import os
import hashlib
import logging
from typing import Any

import gspread
import pandas as pd
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_CREDENTIALS_PATH = os.getenv("GOOGLE_CREDENTIALS_PATH")
SHEET_ID = os.getenv("SHEET_ID")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.readonly",
]

COLLECTION_NAME = "legal_chunks"
VECTOR_SIZE = 384


# ---------------------------------------------------------------------------
# Google Sheets helpers
# ---------------------------------------------------------------------------

def _get_sheets_client() -> gspread.Client:
    creds = Credentials.from_service_account_file(GOOGLE_CREDENTIALS_PATH, scopes=SCOPES)
    return gspread.authorize(creds)


def load_inventory() -> pd.DataFrame:
    log.info("Loading 'Document Inventory' from Google Sheets …")
    try:
        client = _get_sheets_client()
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet("Document Inventory")
        rows = worksheet.get_all_records()
        df = pd.DataFrame(rows)
        log.info("  → %d documents loaded", len(df))
        return df
    except Exception as e:
        log.error("Failed to load Document Inventory: %s", e)
        raise


def load_structure() -> pd.DataFrame:
    log.info("Loading 'Document Structure' from Google Sheets …")
    try:
        client = _get_sheets_client()
        sheet = client.open_by_key(SHEET_ID)
        worksheet = sheet.worksheet("Document Structure")
        rows = worksheet.get_all_records()
        df = pd.DataFrame(rows)
        log.info("  → %d structure rows loaded", len(df))
        return df
    except Exception as e:
        log.error("Failed to load Document Structure: %s", e)
        raise


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def build_chunk_text(row: dict[str, Any]) -> str:
    """Build a self-contained chunk string from one Document Structure row."""
    parts: list[str] = []

    doc_id = str(row.get("doc_id", "")).strip()
    unit_type = str(row.get("unit_type", "")).strip().lower()
    section_number = str(row.get("section_number", "")).strip()
    section_title = str(row.get("section_title", "")).strip()
    business_relevance = str(row.get("business_relevance", "")).strip()
    plain_summary = str(row.get("plain_summary", "")).strip()

    if unit_type == "section":
        label = f"Section {section_number}" if section_number else "Section"
        parts.append(f"[{doc_id}] {label}: {section_title}")
    elif unit_type in ("issue", "holding"):
        label = unit_type.capitalize()
        parts.append(f"[{doc_id}] {label}: {section_title}")
    else:
        parts.append(f"[{doc_id}] {section_title}")

    if business_relevance:
        parts.append(f"Business Relevance: {business_relevance}")
    if plain_summary:
        parts.append(f"Summary: {plain_summary}")

    return "\n".join(parts)


def stable_point_id(doc_id: str, section_number: str) -> int:
    """Deterministic int64 ID from doc_id + section_number."""
    raw = f"{doc_id}::{section_number}"
    digest = hashlib.sha256(raw.encode()).hexdigest()
    return int(digest[:16], 16)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_chunks(texts: list[str]) -> list[list[float]]:
    log.info("Loading embedding model (all-MiniLM-L6-v2) …")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    log.info("  → Encoding %d chunks …", len(texts))
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    log.info("  → Embeddings generated (%d vectors, dim=%d)", len(embeddings), len(embeddings[0]))
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def write_to_qdrant(chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> int:
    log.info("Connecting to Qdrant at localhost:6333 …")
    try:
        client = QdrantClient(url="http://localhost:6333")

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        log.info("  → Collection '%s' ready", COLLECTION_NAME)

        points: list[PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings):
            payload = {
                "doc_id": chunk["doc_id"],
                "domain": chunk.get("domain", ""),
                "doc_type": chunk.get("doc_type", ""),
                "unit_type": chunk.get("unit_type", ""),
                "section_number": chunk.get("section_number", ""),
                "section_title": chunk.get("section_title", ""),
                "relevance_tags": chunk.get("relevance_tags_list", []),
                "business_scenario": chunk.get("business_scenario", ""),
                "chunk_text": chunk["chunk_text"],
            }
            point = PointStruct(
                id=stable_point_id(chunk["doc_id"], chunk.get("section_number", "")),
                vector=embedding,
                payload=payload,
            )
            points.append(point)

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            log.info("  → Upserted points %d–%d", i, min(i + batch_size - 1, len(points) - 1))

        log.info("  → %d points written to Qdrant", len(points))
        return len(points)
    except Exception as e:
        log.error("Qdrant write failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Neo4j
# ---------------------------------------------------------------------------

def write_to_neo4j(inventory_df: pd.DataFrame, structure_df: pd.DataFrame) -> dict[str, int]:
    log.info("Connecting to Neo4j at %s …", NEO4J_URI)
    stats = {"nodes_created": 0, "rels_created": 0, "rows_skipped": 0}

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:

            # ---------- Act / Judgment nodes ----------
            for _, row in inventory_df.iterrows():
                try:
                    doc_id = str(row["doc_id"]).strip()
                    doc_type = str(row.get("doc_type", "")).strip().lower()
                    props: dict[str, Any] = {
                        "doc_id": doc_id,
                        "title": str(row.get("title", "")),
                        "domain": str(row.get("domain", "")),
                        "year": str(row.get("year", "")),
                        "source_url": str(row.get("source_url", "")),
                        "file_path": str(row.get("file_path", "")),
                    }
                    if doc_type == "act":
                        label = "Act"
                        props["act_version"] = str(row.get("act_version", ""))
                        props["status"] = str(row.get("status", ""))
                    else:
                        label = "Judgment"
                        props["court"] = str(row.get("court", ""))
                        props["judgment_status"] = str(row.get("judgment_status", ""))

                    session.run(
                        f"MERGE (d:{label} {{doc_id: $doc_id}}) "
                        "SET d += $props",
                        doc_id=doc_id,
                        props=props,
                    )
                    stats["nodes_created"] += 1
                except Exception as e:
                    log.warning("Skipping inventory row doc_id=%s: %s", row.get("doc_id"), e)
                    stats["rows_skipped"] += 1

            log.info("  → %d Act/Judgment nodes created", stats["nodes_created"])

            # ---------- Domain nodes & HAS_DOMAIN relationships ----------
            domains = inventory_df["domain"].dropna().unique()
            for domain in domains:
                session.run(
                    "MERGE (d:Domain {name: $name})",
                    name=str(domain).strip(),
                )
            session.run(
                "MATCH (n:Act) MATCH (d:Domain {name: n.domain}) MERGE (n)-[:HAS_DOMAIN]->(d)"
            )
            session.run(
                "MATCH (n:Judgment) MATCH (d:Domain {name: n.domain}) MERGE (n)-[:HAS_DOMAIN]->(d)"
            )
            log.info("  → %d Domain nodes ready", len(domains))

            # ---------- Section / Issue nodes ----------
            for _, row in structure_df.iterrows():
                try:
                    doc_id = str(row["doc_id"]).strip()
                    unit_type = str(row.get("unit_type", "")).strip().lower()
                    section_number = str(row.get("section_number", "")).strip()
                    section_title = str(row.get("section_title", "")).strip()
                    page_number = str(row.get("page_number", ""))
                    business_scenario = str(row.get("business_scenario", "")).strip()
                    relevance_tags = str(row.get("relevance_tags", "")).strip()
                    business_relevance = str(row.get("business_relevance", "")).strip()
                    plain_summary = str(row.get("plain_summary", "")).strip()
                    related_doc_ids = str(row.get("related_doc_ids", "")).strip()

                    if unit_type == "section":
                        label = "Section"
                        parent_label = "Act"
                        rel_type = "HAS_SECTION"
                        uid = f"{doc_id}::{section_number}"
                    elif unit_type in ("issue", "holding"):
                        label = "Issue"
                        parent_label = "Judgment"
                        rel_type = "HAS_ISSUE"
                        uid = f"{doc_id}::{section_number}"
                    else:
                        log.warning("Unknown unit_type '%s' for doc_id=%s, skipping", unit_type, doc_id)
                        stats["rows_skipped"] += 1
                        continue

                    tags_list = [t.strip() for t in relevance_tags.split(",") if t.strip()]

                    props = {
                        "uid": uid,
                        "doc_id": doc_id,
                        "section_number": section_number,
                        "section_title": section_title,
                        "page_number": page_number,
                        "business_scenario": business_scenario,
                        "business_relevance": business_relevance,
                        "plain_summary": plain_summary,
                        "relevance_tags": tags_list,
                    }

                    session.run(
                        f"MERGE (s:{label} {{uid: $uid}}) "
                        "SET s += $props",
                        uid=uid,
                        props=props,
                    )
                    stats["nodes_created"] += 1

                    # Link to parent Act / Judgment
                    session.run(
                        f"MATCH (p:{parent_label} {{doc_id: $doc_id}}) "
                        f"MATCH (s:{label} {{uid: $uid}}) "
                        f"MERGE (p)-[:{rel_type}]->(s)",
                        doc_id=doc_id,
                        uid=uid,
                    )
                    stats["rels_created"] += 1

                    # LegalIssue node derived from business_scenario
                    if business_scenario:
                        session.run(
                            "MERGE (li:LegalIssue {name: $name})",
                            name=business_scenario,
                        )
                        session.run(
                            f"MATCH (s:{label} {{uid: $uid}}) "
                            "MATCH (li:LegalIssue {name: $name}) "
                            "MERGE (s)-[:ADDRESSES]->(li)",
                            uid=uid,
                            name=business_scenario,
                        )
                        stats["rels_created"] += 1

                    # INTERPRETS relationships (Judgment → Section)
                    if unit_type in ("issue", "holding") and related_doc_ids:
                        for rid in related_doc_ids.split(","):
                            rid = rid.strip()
                            if not rid:
                                continue
                            session.run(
                                "MATCH (j:Judgment {doc_id: $doc_id}) "
                                "MATCH (a:Act {doc_id: $rid})-[:HAS_SECTION]->(s:Section) "
                                "MERGE (j)-[:INTERPRETS]->(s)",
                                doc_id=doc_id,
                                rid=rid,
                            )
                            stats["rels_created"] += 1

                except Exception as e:
                    log.warning("Skipping structure row doc_id=%s section=%s: %s",
                                row.get("doc_id"), row.get("section_number"), e)
                    stats["rows_skipped"] += 1

            log.info("  → %d total nodes, %d relationships created",
                      stats["nodes_created"], stats["rels_created"])

    except Exception as e:
        log.error("Neo4j write failed: %s", e)
        raise
    finally:
        driver.close()

    return stats


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("=" * 60)
    log.info("  Business Legal Assistant — Ingestion Pipeline")
    log.info("=" * 60)

    # 1. Load data
    inventory_df = load_inventory()
    structure_df = load_structure()

    log.info("\nDocuments:")
    for _, row in inventory_df.iterrows():
        doc_id = row.get("doc_id", "?")
        title = row.get("title", "?")
        n_sections = len(structure_df[structure_df["doc_id"] == doc_id])
        log.info("  • %s — %s  (%d chunks)", doc_id, title, n_sections)

    # 2. Build chunks
    log.info("\nBuilding chunk texts …")
    meta_lookup = inventory_df.set_index("doc_id")[["domain", "doc_type"]].to_dict("index")
    chunks: list[dict[str, Any]] = []
    skipped = 0
    for _, row in structure_df.iterrows():
        try:
            row_dict = row.to_dict()
            doc_id = str(row_dict.get("doc_id", "")).strip()
            meta = meta_lookup.get(doc_id, {})
            chunk_text = build_chunk_text(row_dict)
            tags_raw = str(row_dict.get("relevance_tags", ""))
            tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw and tags_raw.lower() != "nan" else []

            chunks.append({
                "doc_id": doc_id,
                "domain": str(meta.get("domain", "")).strip() if meta.get("domain") else "",
                "doc_type": str(meta.get("doc_type", "")).strip() if meta.get("doc_type") else "",
                "unit_type": str(row_dict.get("unit_type", "")).strip(),
                "section_number": str(row_dict.get("section_number", "")).strip(),
                "section_title": str(row_dict.get("section_title", "")).strip(),
                "business_scenario": str(row_dict.get("business_scenario", "")).strip() if pd.notna(row_dict.get("business_scenario")) else "",
                "relevance_tags_list": tags_list,
                "chunk_text": chunk_text,
            })
        except Exception as e:
            log.warning("Skipping row doc_id=%s: %s", row.get("doc_id"), e)
            skipped += 1

    log.info("  → %d chunks built, %d skipped", len(chunks), skipped)

    # 3. Embed
    texts = [c["chunk_text"] for c in chunks]
    embeddings = embed_chunks(texts)

    # 4. Write to Qdrant
    qdrant_count = write_to_qdrant(chunks, embeddings)

    # 5. Write to Neo4j
    neo4j_stats = write_to_neo4j(inventory_df, structure_df)

    # 6. Summary
    log.info("\n" + "=" * 60)
    log.info("  Pipeline Summary")
    log.info("=" * 60)
    log.info("  Documents processed     : %d", len(inventory_df))
    log.info("  Structure rows processed: %d", len(structure_df))
    log.info("  Chunks embedded         : %d", len(embeddings))
    log.info("  Qdrant points written   : %d", qdrant_count)
    log.info("  Neo4j nodes created     : %d", neo4j_stats["nodes_created"])
    log.info("  Neo4j rels created      : %d", neo4j_stats["rels_created"])
    log.info("  Rows skipped            : %d", neo4j_stats["rows_skipped"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
