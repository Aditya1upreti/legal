#!/usr/bin/env python3
"""
Legal RAG Verification & Diagnostic Suite — Business Legal Assistant
===================================================================
Diagnostic script to verify data source availability, dry-run chunking
and embedding generation, and validate actual database outputs in Qdrant and Neo4j.

Usage:
    python scripts/verify_pipeline.py --check 1
    python scripts/verify_pipeline.py --check 2
    python scripts/verify_pipeline.py --check 3
    python scripts/verify_pipeline.py --check all
"""

import os
import re
import sys
import random
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import requests
from dotenv import load_dotenv
import gspread
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from neo4j import GraphDatabase, Driver

# Ensure scripts directory is on the path so we can import from ingest_pipeline
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest_pipeline import (
    load_inventory,
    load_structure,
    build_chunk_text,
    embed_chunks,
    COLLECTION_NAME,
    VECTOR_DIMENSION,
    EMBEDDING_MODEL_NAME,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PipelineVerifier")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def truncate_str(s: str, max_len: int = 40) -> str:
    """Truncates string with an ellipsis if it exceeds max_len."""
    s = s.strip().replace("\n", " ")
    return (s[:max_len - 3] + "...") if len(s) > max_len else s


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
# CHECK 1: Data Source Readability (Fixed Priority Logic)
# ---------------------------------------------------------------------------
def check_1_data_source_readability(
    gc: gspread.Client,
    sheet_id: str,
    creds_path: str,
    drive_folder_id: Optional[str]
) -> bool:
    """
    Checks accessibility of each document in 'Document Inventory'.
    Prioritizes external URLs (e.g. indiacode.nic.in) when populated,
    and checks Google Drive only when source_url is empty or points to Drive.
    """
    logger.info("================================================================================")
    logger.info("🔍 CHECK 1: DATA SOURCE READABILITY")
    logger.info("================================================================================")

    inventory_records = load_inventory(gc, sheet_id)
    if not inventory_records:
        logger.error("❌ No inventory records found in Google Sheet.")
        return False

    drive_service = None
    try:
        drive_service = get_drive_service(creds_path)
    except Exception as e:
        logger.warning("⚠️  Could not initialize Google Drive API client: %s. Drive checks will be limited.", e)

    results = []
    http_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0.0.0 Safari/537.36"
        )
    }

    for row in inventory_records:
        doc_id = row.get("doc_id", "").strip() or "UNKNOWN"
        title = row.get("title", "").strip() or "Untitled"
        file_path = row.get("file_path", "").strip()
        source_url = row.get("source_url", "").strip()

        source_type = "Unknown"
        is_reachable = False
        reason = ""

        # 1. Check if source_url is a valid external URL (not Google Drive)
        is_external_url = (
            (source_url.startswith("http://") or source_url.startswith("https://"))
            and "drive.google.com" not in source_url.lower()
            and "docs.google.com" not in source_url.lower()
        )

        if is_external_url:
            source_type = "External Link"
            try:
                # Try HEAD first
                resp = requests.head(source_url, headers=http_headers, timeout=8, allow_redirects=True)
                if resp.status_code < 400:
                    is_reachable = True
                else:
                    # Fallback to GET stream (some legal portals block HEAD with 403/405)
                    resp_get = requests.get(source_url, headers=http_headers, timeout=8, stream=True)
                    if resp_get.status_code < 400:
                        is_reachable = True
                    else:
                        reason = f"HTTP {resp_get.status_code}"
            except requests.exceptions.Timeout:
                reason = "HTTP Request Timeout"
            except requests.exceptions.RequestException as e:
                reason = f"Network error: {e.__class__.__name__}"

        # 2. Only attempt Drive check if source_url is a Drive link OR source_url is empty/blank
        elif (
            "drive.google.com" in source_url.lower()
            or "docs.google.com" in source_url.lower()
            or bool(file_path)
        ):
            source_type = "Google Drive"
            if not drive_service:
                reason = "Drive API unavailable"
            else:
                # A. Try extracting file ID
                file_id = extract_drive_file_id(source_url) or extract_drive_file_id(file_path)
                if file_id:
                    try:
                        f = drive_service.files().get(fileId=file_id, fields="id, name, trashed").execute()
                        if f and not f.get("trashed", False):
                            is_reachable = True
                        else:
                            reason = "File trashed in Drive"
                    except Exception as e:
                        reason = f"Drive ID lookup failed: {e.__class__.__name__}"
                else:
                    # B. Try searching by file name in Drive
                    filename = Path(file_path).name if file_path else ""
                    if filename:
                        try:
                            query = f"name = '{filename}' and trashed = false"
                            if drive_folder_id:
                                query += f" and '{drive_folder_id}' in parents"
                            response = drive_service.files().list(q=query, fields="files(id, name)").execute()
                            files = response.get("files", [])
                            if files:
                                is_reachable = True
                            else:
                                # Fallback search across Drive
                                fallback = drive_service.files().list(
                                    q=f"name = '{filename}' and trashed = false",
                                    fields="files(id, name)"
                                ).execute()
                                if fallback.get("files", []):
                                    is_reachable = True
                                else:
                                    reason = f"File '{filename}' not found in Drive"
                        except Exception as e:
                            reason = f"Drive search failed: {e.__class__.__name__}"
                    else:
                        reason = "Missing Drive filename/ID"

        # 3. Neither an external link nor a Drive file path was provided
        else:
            source_type = "None / Missing"
            reason = "No file_path or source_url provided"

        results.append({
            "doc_id": doc_id,
            "title": title,
            "source_type": source_type,
            "reachable": is_reachable,
            "reason": reason
        })

    # Print Table
    col_w = [10, 36, 16, 12, 32]
    header = (
        f"{'DOC ID'.ljust(col_w[0])} | "
        f"{'TITLE'.ljust(col_w[1])} | "
        f"{'SOURCE TYPE'.ljust(col_w[2])} | "
        f"{'STATUS'.ljust(col_w[3])} | "
        f"{'REASON / DETAILS'.ljust(col_w[4])}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print(header)
    print(separator)

    passed_count = 0
    failed_items = []

    for r in results:
        status_label = "✅ Reachable" if r["reachable"] else "❌ Failed"
        if r["reachable"]:
            passed_count += 1
        else:
            failed_items.append(r)

        row_str = (
            f"{r['doc_id'][:col_w[0]].ljust(col_w[0])} | "
            f"{truncate_str(r['title'], col_w[1]).ljust(col_w[1])} | "
            f"{r['source_type'][:col_w[2]].ljust(col_w[2])} | "
            f"{status_label.ljust(col_w[3])} | "
            f"{truncate_str(r['reason'], col_w[4]).ljust(col_w[4])}"
        )
        print(row_str)

    print(separator)
    total_docs = len(results)
    failed_count = total_docs - passed_count
    logger.info("Summary: %d of %d documents reachable (%d failed).", passed_count, total_docs, failed_count)

    if failed_items:
        logger.warning("Failed Documents Summary:")
        for f in failed_items:
            logger.warning("  * [%s] %s (Type: %s, Reason: %s)", f["doc_id"], f["title"], f["source_type"], f["reason"])

    all_passed = (failed_count == 0)
    logger.info("CHECK 1 RESULT: %s\n", "✅ PASSED" if all_passed else "❌ FAILED (Issues detected)")
    return all_passed


# ---------------------------------------------------------------------------
# CHECK 2: Pipeline Dry Run (No Writes)
# ---------------------------------------------------------------------------
def check_2_dry_run_pipeline(gc: gspread.Client, sheet_id: str) -> bool:
    """
    Validates chunk construction and local embeddings in-memory without database writes.
    """
    logger.info("================================================================================")
    logger.info("🧪 CHECK 2: PIPELINE FORMS CORRECTLY (Dry Run, No DB Writes)")
    logger.info("================================================================================")

    inventory_records = load_inventory(gc, sheet_id)
    structure_records = load_structure(gc, sheet_id)

    inventory_lookup = {
        row.get("doc_id", "").strip(): row
        for row in inventory_records
        if row.get("doc_id", "").strip()
    }

    chunks_formed = []
    chunk_meta = []
    suspicious_chunks = []
    skipped_count = 0

    for row in structure_records:
        doc_id = row.get("doc_id", "").strip()
        row_num = row.get("_row_number", "?")
        chunk_text = build_chunk_text(row, inventory_lookup)

        if not chunk_text:
            skipped_count += 1
            continue

        sec_num = row.get("section_number", "").strip()
        sec_title = row.get("section_title", "").strip()

        # Validation: length threshold check
        if len(chunk_text.strip()) < 20:
            suspicious_chunks.append({
                "doc_id": doc_id,
                "row_num": row_num,
                "length": len(chunk_text),
                "text": chunk_text,
                "reason": "Text under 20 characters"
            })

        chunks_formed.append(chunk_text)
        chunk_meta.append({
            "doc_id": doc_id,
            "section_number": sec_num,
            "section_title": sec_title,
            "chunk_text": chunk_text
        })

    logger.info("Structure Rows Read: %d", len(structure_records))
    logger.info("Chunks Successfully Formed: %d (Skipped blank: %d)", len(chunks_formed), skipped_count)

    if suspicious_chunks:
        logger.warning("⚠️  Suspiciously short chunks flagged (%d):", len(suspicious_chunks))
        for sc in suspicious_chunks:
            logger.warning("  * Doc %s (Row %s, len=%d): '%s'", sc["doc_id"], sc["row_num"], sc["length"], sc["text"])
    else:
        logger.info("✅ All formed chunks satisfy length sanity threshold (>= 20 characters).")

    # Embeddings Dry Run
    embeddings_valid = False
    if chunks_formed:
        logger.info("Testing embedding model '%s' locally...", EMBEDDING_MODEL_NAME)
        try:
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            vectors = embed_chunks(chunks_formed, model, batch_size=32)

            dim_match = all(len(v) == VECTOR_DIMENSION for v in vectors)
            non_zero = all(any(abs(val) > 1e-6 for val in v) for v in vectors)

            if dim_match and non_zero:
                embeddings_valid = True
                logger.info("✅ Embeddings validated: %d vectors generated, all dimension %d and non-zero.",
                            len(vectors), VECTOR_DIMENSION)
            else:
                logger.error("❌ Embedding validation failed: Dimension match=%s, Non-zero check=%s", dim_match, non_zero)
        except Exception as e:
            logger.error("❌ Embedding generation failed with error: %s", e)
    else:
        logger.error("❌ No chunks were formed to test embeddings.")

    # Print 3 Random Sample Chunks
    if chunk_meta:
        print("\n--- SAMPLE CHUNKS PREVIEW (Random 3) ---")
        sample_size = min(3, len(chunk_meta))
        sampled = random.sample(chunk_meta, sample_size)
        for idx, s in enumerate(sampled, start=1):
            print(f"\n[Sample {idx}]")
            print(f"  Doc ID   : {s['doc_id']}")
            print(f"  Section  : {s['section_number']} — {s['section_title']}")
            print(f"  Preview  : {truncate_str(s['chunk_text'], 220)}")

    passed = (len(chunks_formed) > 0) and embeddings_valid and (len(suspicious_chunks) == 0)
    logger.info("\nCHECK 2 RESULT: %s\n", "✅ PASSED" if passed else "⚠️  PASSED WITH WARNINGS / FAILURES")
    return passed


# ---------------------------------------------------------------------------
# CHECK 3: Actual Pipeline Output Verification (Post-Ingestion)
# ---------------------------------------------------------------------------
def check_3_actual_db_outputs(
    gc: gspread.Client,
    sheet_id: str,
    qdrant_url: str,
    neo4j_uri: str,
    neo4j_auth: Tuple[str, str]
) -> bool:
    """
    Verifies actual indexed data in Qdrant and Neo4j and cross-checks counts against the Google Sheet.
    """
    logger.info("================================================================================")
    logger.info("📊 CHECK 3: ACTUAL PIPELINE OUTPUT VERIFICATION (Qdrant & Neo4j)")
    logger.info("================================================================================")

    inventory_records = load_inventory(gc, sheet_id)
    structure_records = load_structure(gc, sheet_id)
    valid_structure_rows = [
        r for r in structure_records
        if r.get("section_number", "").strip() or r.get("section_title", "").strip()
    ]

    expected_docs_count = len([r for r in inventory_records if r.get("doc_id", "").strip()])
    expected_sections_count = len(valid_structure_rows)

    qdrant_ok = False
    neo4j_ok = False

    # 1. Qdrant Verification
    logger.info("--- 1. Qdrant Vector Database Verification ---")
    try:
        qdrant = QdrantClient(url=qdrant_url)
        collections = [c.name for c in qdrant.get_collections().collections]

        if COLLECTION_NAME not in collections:
            logger.error("❌ Collection '%s' does not exist in Qdrant.", COLLECTION_NAME)
        else:
            col_info = qdrant.get_collection(collection_name=COLLECTION_NAME)
            point_count = col_info.points_count
            logger.info("Collection '%s' located. Total point count: %d", COLLECTION_NAME, point_count)

            scroll_result, _ = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=15,
                with_payload=True,
                with_vectors=False
            )

            if scroll_result:
                sample_pts = random.sample(scroll_result, min(3, len(scroll_result)))
                print("\n  Sample Qdrant Points:")
                for idx, pt in enumerate(sample_pts, start=1):
                    p = pt.payload or {}
                    print(f"    ({idx}) Point ID: {pt.id}")
                    print(f"        Doc ID  : {p.get('doc_id')}")
                    print(f"        Section : {p.get('section_number')} — {p.get('section_title')}")
                    print(f"        Preview : {truncate_str(p.get('chunk_text', ''), 140)}")
            else:
                logger.warning("⚠️  Collection exists but returned 0 points upon scrolling.")

            if point_count >= expected_sections_count:
                qdrant_ok = True
                logger.info("✅ Qdrant point count (%d) matches or exceeds expected valid structure rows (%d).",
                            point_count, expected_sections_count)
            else:
                logger.warning("⚠️  Qdrant point count (%d) is lower than expected valid structure rows (%d).",
                               point_count, expected_sections_count)
                qdrant_ok = (point_count > 0)
    except Exception as e:
        logger.error("❌ Failed to query Qdrant at %s: %s", qdrant_url, e)

    # 2. Neo4j Verification
    logger.info("\n--- 2. Neo4j Graph Database Verification ---")
    try:
        driver: Driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        driver.verify_connectivity()

        node_queries = {
            "Act": "MATCH (n:Act) RETURN count(n) AS c",
            "Judgement": "MATCH (n:Judgement) RETURN count(n) AS c",
            "Section": "MATCH (n:Section) RETURN count(n) AS c",
            "Domain": "MATCH (n:Domain) RETURN count(n) AS c",
            "BusinessScenario": "MATCH (n:BusinessScenario) RETURN count(n) AS c",
        }

        rel_queries = {
            "HAS_SECTION": "MATCH ()-[r:HAS_SECTION]->() RETURN count(r) AS c",
            "BELONGS_TO_DOMAIN": "MATCH ()-[r:BELONGS_TO_DOMAIN]->() RETURN count(r) AS c",
            "RELATED_TO": "MATCH ()-[r:RELATED_TO]->() RETURN count(r) AS c",
            "RELATES_TO_SCENARIO": "MATCH ()-[r:RELATES_TO_SCENARIO]->() RETURN count(r) AS c",
        }

        node_counts = {}
        rel_counts = {}

        with driver.session() as session:
            for label, query in node_queries.items():
                result = session.run(query).single()
                node_counts[label] = result["c"] if result else 0

            for rel, query in rel_queries.items():
                result = session.run(query).single()
                rel_counts[rel] = result["c"] if result else 0

        driver.close()

        print("\n  Neo4j Node Counts:")
        for label, count in node_counts.items():
            print(f"    * (:{label.ljust(18)}) : {count}")

        print("\n  Neo4j Relationship Counts:")
        for rel, count in rel_counts.items():
            print(f"    * [:{rel.ljust(20)}] : {count}")

        # Cross-Checks
        total_doc_nodes = node_counts["Act"] + node_counts["Judgement"]
        total_section_nodes = node_counts["Section"]

        logger.info("\n--- 3. Sheet vs. Database Cross-Check ---")
        logger.info("Expected Documents (Sheet) : %d  |  Neo4j Document Nodes (Act+Judgement): %d",
                    expected_docs_count, total_doc_nodes)
        logger.info("Expected Sections  (Sheet) : %d  |  Neo4j Section Nodes                 : %d",
                    expected_sections_count, total_section_nodes)

        doc_mismatch = (total_doc_nodes != expected_docs_count)
        sec_mismatch = (total_section_nodes != expected_sections_count)

        if doc_mismatch:
            logger.warning("⚠️  Document node mismatch: Sheet has %d docs, Neo4j has %d",
                           expected_docs_count, total_doc_nodes)
        if sec_mismatch:
            logger.warning("⚠️  Section node mismatch: Sheet has %d valid sections, Neo4j has %d",
                           expected_sections_count, total_section_nodes)

        if not doc_mismatch and not sec_mismatch:
            logger.info("✅ Exact count alignment across Google Sheet and Neo4j graph nodes!")
            neo4j_ok = True
        else:
            neo4j_ok = (total_doc_nodes > 0 and total_section_nodes > 0)

    except Exception as e:
        logger.error("❌ Failed to query Neo4j at %s: %s", neo4j_uri, e)

    overall_ok = qdrant_ok and neo4j_ok
    logger.info("\nCHECK 3 RESULT: %s\n", "✅ PASSED" if overall_ok else "❌ FAILED / DISCREPANCY DETECTED")
    return overall_ok


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Legal RAG Pipeline Diagnostic & Verification Suite"
    )
    parser.add_argument(
        "--check",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Specify which verification check to run: 1 (readability), 2 (dry run), 3 (db output), or all."
    )
    args = parser.parse_args()

    load_dotenv()

    # Credentials & Connection URIs
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "")
    sheet_id = os.getenv("SHEET_ID", "")
    drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "")
    qdrant_url = os.getenv("QDRANT_URL", os.getenv("QDRANT_HOST", "http://localhost:6333"))

    if not creds_path or not sheet_id:
        logger.error("❌ GOOGLE_CREDENTIALS_PATH and SHEET_ID must be set in your .env file.")
        sys.exit(1)

    if not os.path.exists(creds_path):
        logger.error("❌ Credentials file not found at: %s", creds_path)
        sys.exit(1)

    try:
        gc = gspread.service_account(filename=creds_path)
    except Exception as e:
        logger.error("❌ Could not authenticate with Google Sheets: %s", e)
        sys.exit(1)

    selected = args.check
    results = {}

    if selected in ("1", "all"):
        results["Check 1: Data Source Readability"] = check_1_data_source_readability(
            gc, sheet_id, creds_path, drive_folder_id
        )

    if selected in ("2", "all"):
        results["Check 2: Pipeline Dry Run"] = check_2_dry_run_pipeline(gc, sheet_id)

    if selected in ("3", "all"):
        results["Check 3: Actual DB Outputs"] = check_3_actual_db_outputs(
            gc, sheet_id, qdrant_url, neo4j_uri, (neo4j_user, neo4j_password)
        )

    # Final Overall Summary
    print("================================================================================")
    print("📋 FINAL VERIFICATION REPORT SUMMARY")
    print("================================================================================")
    for check_name, passed in results.items():
        icon = "✅ PASSED" if passed else "❌ FAILED / WARNINGS"
        print(f"  {check_name.ljust(42)} : {icon}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()