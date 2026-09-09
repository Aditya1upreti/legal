#!/usr/bin/env python3
"""
Legal RAG PDF Extraction & Content Verifier — Business Legal Assistant
=====================================================================
Downloads referenced legal PDFs into memory (from external URLs or Google Drive),
extracts text using PyMuPDF (fitz), inspects page counts and character lengths,
and cross-checks section numbers against the Document Structure tab.

Usage:
    python scripts/verify_pdf_reading.py
"""

import io
import os
import re
import sys
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

import requests
from dotenv import load_dotenv
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Check for PyMuPDF (fitz)
try:
    import fitz
except ImportError:
    print(
        "\n[ERROR] PyMuPDF is not installed. Please install it in your virtual environment via:\n"
        "    pip install pymupdf\n",
        file=sys.stderr
    )
    sys.exit(1)

# Ensure project root & scripts directory are on sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest_pipeline import load_inventory, load_structure

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PDFVerifier")

# Minimum character count to flag possible scanned / image-only PDFs
MIN_CHAR_COUNT_THRESHOLD = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_text_preview(text: str, max_chars: int = 300) -> str:
    """Normalizes whitespace and returns a truncated text preview."""
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) > max_chars:
        return clean[:max_chars - 3] + "..."
    return clean


def parse_page_number_int(val: Any) -> Optional[int]:
    """Safely extracts integer page count from sheet cell string/int."""
    if val is None:
        return None
    match = re.search(r"\d+", str(val))
    return int(match.group(0)) if match else None


def extract_drive_file_id(path_or_url: str) -> Optional[str]:
    """Extracts a Google Drive file ID from a URL or raw ID string."""
    match = re.search(r"/d/([a-zA-Z0-9_-]{25,})", path_or_url)
    if match:
        return match.group(1)
    match = re.search(r"id=([a-zA-Z0-9_-]{25,})", path_or_url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{25,50}", path_or_url.strip()):
        return path_or_url.strip()
    return None


def get_drive_service(creds_path: str):
    """Initializes Google Drive API service client."""
    scopes = ["https://www.googleapis.com/auth/drive.readonly"]
    creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


# ---------------------------------------------------------------------------
# In-Memory Fetchers
# ---------------------------------------------------------------------------
def fetch_external_url_bytes(url: str, timeout: int = 25) -> bytes:
    """Downloads remote PDF into memory buffer without saving to disk."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*"
    }
    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()
    return resp.content


def fetch_drive_file_bytes(
    drive_service,
    file_id: Optional[str] = None,
    filename: Optional[str] = None,
    drive_folder_id: Optional[str] = None
) -> bytes:
    """Downloads Google Drive file into an in-memory BytesIO buffer."""
    target_id = file_id

    # If no file_id provided, look up by filename in Drive
    if not target_id and filename:
        query = f"name = '{filename}' and trashed = false"
        if drive_folder_id:
            query += f" and '{drive_folder_id}' in parents"
        response = drive_service.files().list(q=query, fields="files(id, name)").execute()
        files = response.get("files", [])
        if files:
            target_id = files[0]["id"]
        else:
            # Fallback search anywhere in accessible drive
            fallback = drive_service.files().list(
                q=f"name = '{filename}' and trashed = false",
                fields="files(id, name)"
            ).execute()
            f_files = fallback.get("files", [])
            if f_files:
                target_id = f_files[0]["id"]

    if not target_id:
        raise ValueError(f"Could not locate file in Google Drive (name: {filename}, id: {file_id})")

    request = drive_service.files().get_media(fileId=target_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# PDF Extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> Tuple[int, str]:
    """
    Opens PDF bytes using PyMuPDF (fitz) and extracts text across all pages.
    Returns (page_count, full_extracted_text).
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = len(doc)
        text_parts = []
        for page_num in range(page_count):
            page = doc.load_page(page_num)
            text_parts.append(page.get_text())
        full_text = "\n".join(text_parts)
        return page_count, full_text


# ---------------------------------------------------------------------------
# Main Verification Logic
# ---------------------------------------------------------------------------
def main():
    load_dotenv()

    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "")
    sheet_id = os.getenv("SHEET_ID", "")
    drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "")

    if not creds_path or not sheet_id:
        logger.error("❌ GOOGLE_CREDENTIALS_PATH and SHEET_ID must be set in your .env file.")
        sys.exit(1)

    if not os.path.exists(creds_path):
        logger.error("❌ Credentials file not found at: %s", creds_path)
        sys.exit(1)

    logger.info("Initializing Google Sheets and Drive connections...")
    try:
        gc = gspread.service_account(filename=creds_path)
        drive_service = get_drive_service(creds_path)
    except Exception as e:
        logger.error("❌ Initialization failure: %s", e)
        sys.exit(1)

    # 1. Load Sheets Data
    inventory_records = load_inventory(gc, sheet_id)
    structure_records = load_structure(gc, sheet_id)

    # Group structure rows by doc_id for sampling and cross-check
    sections_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in structure_records:
        d_id = row.get("doc_id", "").strip()
        if d_id:
            sections_by_doc[d_id].append(row)

    logger.info("================================================================================")
    logger.info("📄 PDF EXTRACTION & TEXT READABILITY VERIFICATION")
    logger.info("================================================================================")

    doc_results = []
    extracted_texts: Dict[str, str] = {}

    for idx, doc in enumerate(inventory_records, start=1):
        doc_id = doc.get("doc_id", "").strip() or f"DOC_{idx}"
        title = doc.get("title", "").strip() or "Untitled"
        file_path = doc.get("file_path", "").strip()
        source_url = doc.get("source_url", "").strip()
        sheet_pages = parse_page_number_int(doc.get("pages"))

        logger.info("--------------------------------------------------------------------------------")
        logger.info("[%d/%d] Testing Doc: %s — '%s'", idx, len(inventory_records), doc_id, title)

        is_external = (
            (source_url.startswith("http://") or source_url.startswith("https://"))
            and "drive.google.com" not in source_url.lower()
            and "docs.google.com" not in source_url.lower()
        )

        pdf_bytes = None
        source_desc = ""
        fetch_error = None

        # A. Fetch Bytes
        try:
            if is_external:
                source_desc = f"External URL ({source_url})"
                pdf_bytes = fetch_external_url_bytes(source_url)
            elif "drive.google.com" in source_url.lower() or "docs.google.com" in source_url.lower() or file_path:
                source_desc = f"Google Drive ({file_path or source_url})"
                file_id = extract_drive_file_id(source_url) or extract_drive_file_id(file_path)
                filename = Path(file_path).name if file_path else None
                pdf_bytes = fetch_drive_file_bytes(drive_service, file_id=file_id, filename=filename, drive_folder_id=drive_folder_id)
            else:
                raise ValueError("No valid file_path or source_url provided.")
        except Exception as e:
            fetch_error = str(e)
            logger.error("  ❌ Download failed from %s: %s", source_desc, e)

        # B. Parse PDF with PyMuPDF
        opened_ok = False
        actual_pages = 0
        char_count = 0
        preview = ""
        parse_error = None

        if pdf_bytes:
            try:
                actual_pages, full_text = extract_text_from_pdf_bytes(pdf_bytes)
                opened_ok = True
                char_count = len(full_text)
                preview = clean_text_preview(full_text, max_chars=300)
                extracted_texts[doc_id] = full_text

                # Page count comparison
                page_match_msg = f"{actual_pages} pages"
                if sheet_pages is not None:
                    if sheet_pages == actual_pages:
                        page_match_msg += f" (✅ matches sheet: {sheet_pages})"
                    else:
                        page_match_msg += f" (⚠️  sheet specifies {sheet_pages} pages)"

                logger.info("  ✅ Opened PDF successfully: %s, %d characters extracted.", page_match_msg, char_count)

                # Character length sanity check
                if char_count < MIN_CHAR_COUNT_THRESHOLD:
                    logger.warning("  ⚠️  LOW TEXT WARNING: Extracted only %d characters. (Possible scanned image-only PDF needing OCR)", char_count)
                else:
                    logger.info("  Preview: \"%s\"", preview)

            except Exception as e:
                parse_error = str(e)
                logger.error("  ❌ PyMuPDF parsing failed: %s", e)

        doc_results.append({
            "doc_id": doc_id,
            "title": title,
            "source_desc": source_desc,
            "opened": opened_ok,
            "sheet_pages": sheet_pages,
            "actual_pages": actual_pages,
            "char_count": char_count,
            "preview": preview,
            "error": fetch_error or parse_error
        })

    # ---------------------------------------------------------------------------
    # Cross-Check Sample (3 Random Docs vs Structure Section Numbers)
    # ---------------------------------------------------------------------------
    logger.info("\n================================================================================")
    logger.info("🔬 CROSS-CHECK: VERIFYING SECTION NUMBERS IN EXTRACTED PDF TEXT")
    logger.info("================================================================================")

    eligible_docs = [
        d["doc_id"] for d in doc_results
        if d["opened"] and d["char_count"] >= MIN_CHAR_COUNT_THRESHOLD and len(sections_by_doc.get(d["doc_id"], [])) > 0
    ]

    cross_check_results = []
    if eligible_docs:
        sample_size = min(3, len(eligible_docs))
        sampled_doc_ids = random.sample(eligible_docs, sample_size)

        for s_doc_id in sampled_doc_ids:
            doc_sections = sections_by_doc[s_doc_id]
            # Pick a section that has a populated section_number
            valid_sec_rows = [s for s in doc_sections if s.get("section_number", "").strip()]
            if not valid_sec_rows:
                continue

            chosen_row = random.choice(valid_sec_rows)
            sec_num = chosen_row.get("section_number", "").strip()
            sec_title = chosen_row.get("section_title", "").strip()
            pdf_text = extracted_texts.get(s_doc_id, "")

            # Search strategy:
            # 1. Exact match
            # 2. Case-insensitive substring match
            # 3. Numeric anchor search (e.g., "73." or "Section 73")
            matched = False
            if sec_num.lower() in pdf_text.lower():
                matched = True
            else:
                # Try finding bare number if format is "Section N" or "Holding N"
                num_match = re.search(r"\d+", sec_num)
                if num_match:
                    num_val = num_match.group(0)
                    pattern = rf"\b(section|sec\.?|holding)?\s*{num_val}\b"
                    if re.search(pattern, pdf_text, re.IGNORECASE):
                        matched = True

            cross_check_results.append({
                "doc_id": s_doc_id,
                "section_number": sec_num,
                "section_title": sec_title,
                "matched": matched
            })

            if matched:
                logger.info("  ✅ [%s] Target '%s' ('%s') FOUND in extracted PDF text.",
                            s_doc_id, sec_num, truncate_str(sec_title, 35))
            else:
                logger.warning("  ❌ [%s] Target '%s' ('%s') NOT found in extracted PDF text.",
                               s_doc_id, sec_num, truncate_str(sec_title, 35))
    else:
        logger.warning("No eligible documents with sufficient text available to perform cross-check.")

    # ---------------------------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------------------------
    total_docs = len(doc_results)
    opened_count = sum(1 for d in doc_results if d["opened"])
    low_text_count = sum(1 for d in doc_results if d["opened"] and d["char_count"] < MIN_CHAR_COUNT_THRESHOLD)
    failed_count = total_docs - opened_count
    cross_check_failed = [c for c in cross_check_results if not c["matched"]]

    print("\n================================================================================")
    print("📋 PDF EXTRACTION & VERIFICATION SUMMARY REPORT")
    print("================================================================================")
    print(f"Total Documents Tested               : {total_docs}")
    print(f"PDFs Downloaded & Opened Successfully : {opened_count}/{total_docs} ({'✅ ALL PASSED' if opened_count == total_docs else '⚠️ SOME FAILED'})")
    print(f"Suspiciously Low Text (<{MIN_CHAR_COUNT_THRESHOLD} chars)     : {low_text_count} (Potential scanned image PDFs needing OCR)")
    print(f"Failed Documents (Download/Parse Err): {failed_count}")
    print(f"Section Cross-Check Sample Passed     : {len(cross_check_results) - len(cross_check_failed)}/{len(cross_check_results)}")

    if failed_count > 0:
        print("\n--- Failed Documents List ---")
        for d in doc_results:
            if not d["opened"]:
                print(f"  * [{d['doc_id']}] {d['title']} — Error: {d['error']}")

    if low_text_count > 0:
        print("\n--- Suspicious / Low Text Documents List ---")
        for d in doc_results:
            if d["opened"] and d["char_count"] < MIN_CHAR_COUNT_THRESHOLD:
                print(f"  * [{d['doc_id']}] {d['title']} — Extracted only {d['char_count']} chars")

    if cross_check_failed:
        print("\n--- Cross-Check Failures ---")
        for c in cross_check_failed:
            print(f"  * [{c['doc_id']}] Section '{c['section_number']}' not located in PDF text.")

    page_mismatches = [
        d for d in doc_results
        if d["opened"] and d["sheet_pages"] is not None and d["sheet_pages"] != d["actual_pages"]
    ]
    if page_mismatches:
        print("\n--- Page Count Warnings ---")
        for d in page_mismatches:
            print(f"  * [{d['doc_id']}] Sheet specifies {d['sheet_pages']} pages, but PDF has {d['actual_pages']} pages.")

    print("================================================================================\n")


def truncate_str(s: str, max_len: int = 40) -> str:
    """Helper to truncate preview strings cleanly."""
    s = s.strip().replace("\n", " ")
    return (s[:max_len - 3] + "...") if len(s) > max_len else s


if __name__ == "__main__":
    main()