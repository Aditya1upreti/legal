"""
Legal RAG Ingestion Pipeline — Business Legal Assistant
======================================================
Reads structured legal inventory and section data from Google Sheets,
generates dense vector embeddings using a local SentenceTransformer model,
and indexes the resulting chunks and graph topology into Qdrant and Neo4j.
"""

import os
import re
import sys
import uuid
import logging
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict

from dotenv import load_dotenv
import gspread
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from neo4j import GraphDatabase, Driver

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LegalIngestionPipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "legal_chunks"
VECTOR_DIMENSION = 384
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
UUID_NAMESPACE = uuid.NAMESPACE_DNS

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def normalize_header(header: str) -> str:
    """
    Normalizes sheet column headers defensively:
    - Strips whitespace
    - Converts to lowercase
    - Removes parenthetical remarks like '(ACTS ONLY)'
    - Replaces spaces and hyphens with underscores
    """
    h = header.strip().lower()
    h = re.sub(r'\(.*?\)', '', h).strip()
    h = re.sub(r'[\s\-]+', '_', h)
    h = re.sub(r'[^a-z0-9_]', '', h)
    return h


def parse_delimited_list(value: Optional[str], delimiter: str = ",") -> List[str]:
    """Splits a delimited string and strips whitespace from elements."""
    if not value:
        return []
    return [item.strip() for item in str(value).split(delimiter) if item.strip()]


def get_deterministic_uuid(doc_id: str, section_number: str, section_title: str) -> str:
    """Generates a reproducible UUID5 identifier for Qdrant points."""
    key = f"{doc_id.strip()}::{section_number.strip()}::{section_title.strip()}"
    return str(uuid.uuid5(UUID_NAMESPACE, key))


# ---------------------------------------------------------------------------
# Data Loading Functions (Google Sheets)
# ---------------------------------------------------------------------------
def _load_worksheet_records(gc: gspread.Client, sheet_id: str, tab_name: str) -> List[Dict[str, Any]]:
    """Generic helper to read a worksheet and normalize its column headers."""
    try:
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.worksheet(tab_name)
        all_values = worksheet.get_all_values()
        
        if not all_values:
            logger.warning("Worksheet '%s' is completely empty.", tab_name)
            return []

        raw_headers = all_values[0]
        normalized_headers = [normalize_header(h) for h in raw_headers]

        # Defensive handling: ensure first column is treated as doc_id if header is missing
        if normalized_headers and not normalized_headers[0]:
            normalized_headers[0] = "doc_id"

        records = []
        for row_idx, row_values in enumerate(all_values[1:], start=2):
            if not any(v.strip() for v in row_values):
                continue  # Skip trailing empty rows

            record = {}
            for col_idx, col_name in enumerate(normalized_headers):
                if not col_name:
                    continue
                val = row_values[col_idx].strip() if col_idx < len(row_values) else ""
                record[col_name] = val

            record["_row_number"] = row_idx
            records.append(record)

        return records

    except gspread.exceptions.WorksheetNotFound:
        logger.error("Tab '%s' not found in Google Sheet '%s'.", tab_name, sheet_id)
        raise
    except Exception as e:
        logger.error("Failed to load records from worksheet '%s': %s", tab_name, e)
        raise


def load_inventory(gc: gspread.Client, sheet_id: str) -> List[Dict[str, Any]]:
    """Loads Tab 1: Document Inventory."""
    logger.info("Loading 'Document Inventory' tab...")
    records = _load_worksheet_records(gc, sheet_id, "Document Inventory")
    logger.info("Loaded %d documents from inventory.", len(records))
    return records


def load_structure(gc: gspread.Client, sheet_id: str) -> List[Dict[str, Any]]:
    """Loads Tab 2: Document Structure."""
    logger.info("Loading 'Document Structure' tab...")
    records = _load_worksheet_records(gc, sheet_id, "Document Structure")
    logger.info("Loaded %d structure rows.", len(records))
    return records


def load_tag_vocabulary(gc: gspread.Client, sheet_id: str) -> Set[str]:
    """Loads Tab 3: Tag Vocabulary reference set."""
    logger.info("Loading 'Tag Vocabulary' tab...")
    try:
        records = _load_worksheet_records(gc, sheet_id, "Tag Vocabulary")
        valid_tags = {r.get("tag", "").strip() for r in records if r.get("tag", "").strip()}
        logger.info("Loaded %d valid tags into reference vocabulary.", len(valid_tags))
        return valid_tags
    except Exception as e:
        logger.warning("Could not load Tag Vocabulary tab (%s). Soft validation skipped.", e)
        return set()


def load_business_scenarios(gc: gspread.Client, sheet_id: str) -> Set[str]:
    """Loads Tab 4: Business Scenarios reference set."""
    logger.info("Loading 'Business Scenarios' tab...")
    try:
        records = _load_worksheet_records(gc, sheet_id, "Business Scenarios")
        valid_scenarios = {r.get("scenario", "").strip() for r in records if r.get("scenario", "").strip()}
        logger.info("Loaded %d valid business scenarios into reference vocabulary.", len(valid_scenarios))
        return valid_scenarios
    except Exception as e:
        logger.warning("Could not load Business Scenarios tab (%s). Soft validation skipped.", e)
        return set()


# ---------------------------------------------------------------------------
# Chunking and Text Construction
# ---------------------------------------------------------------------------
def build_chunk_text(row: Dict[str, Any], inventory_lookup: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """
    Constructs a clear, self-contained text paragraph for embedding.
    Returns None if both section_number and section_title are empty.
    """
    doc_id = row.get("doc_id", "").strip()
    sec_num = row.get("section_number", "").strip()
    sec_title = row.get("section_title", "").strip()
    
    if not sec_num and not sec_title:
        return None

    parent_doc = inventory_lookup.get(doc_id, {})
    doc_title = parent_doc.get("title", "").strip() or f"Document {doc_id}"
    chapter_title = row.get("chapter_title", "").strip()
    business_relevance = row.get("business_relevance", "").strip()
    plain_summary = row.get("plain_summary", "").strip()

    # Build Header/Context
    prefix_elements = [f"From: {doc_title}"]
    if chapter_title:
        prefix_elements.append(f"— Chapter: {chapter_title}")

    sec_label_parts = []
    if sec_num:
        sec_label_parts.append(sec_num)
    if sec_title:
        sec_label_parts.append(sec_title)

    if sec_label_parts:
        prefix_elements.append(f"— {': '.join(sec_label_parts)}")

    header_text = " ".join(prefix_elements).strip()

    # Build Body Content
    body_elements = []
    if business_relevance:
        rel_str = business_relevance if business_relevance.endswith(('.', '!', '?')) else f"{business_relevance}."
        body_elements.append(rel_str)
    if plain_summary:
        body_elements.append(plain_summary)

    body_text = " ".join(body_elements).strip()

    if header_text and body_text:
        return f"{header_text}. {body_text}"
    return header_text or body_text


# ---------------------------------------------------------------------------
# Local Embeddings
# ---------------------------------------------------------------------------
def embed_chunks(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> List[List[float]]:
    """Encodes a list of chunk strings into dense vectors in efficient batches."""
    logger.info("Generating embeddings for %d chunks using '%s' (batch_size=%d)...", 
                len(texts), EMBEDDING_MODEL_NAME, batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Qdrant Vector Store Ingestion
# ---------------------------------------------------------------------------
def write_to_qdrant(client: QdrantClient, chunks_with_embeddings: List[Dict[str, Any]]) -> int:
    """Ensures collection exists and upserts vectorized chunk points into Qdrant."""
    logger.info("Writing chunks to Qdrant collection '%s'...", COLLECTION_NAME)
    
    # Verify or create collection
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            logger.info("Collection '%s' does not exist. Creating collection...", COLLECTION_NAME)
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=models.Distance.COSINE
                )
            )
    except Exception as e:
        logger.error("Failed to verify or create Qdrant collection '%s': %s", COLLECTION_NAME, e)
        raise

    points = []
    for item in chunks_with_embeddings:
        point = models.PointStruct(
            id=item["point_id"],
            vector=item["vector"],
            payload=item["payload"]
        )
        points.append(point)

    if not points:
        logger.warning("No points to write to Qdrant.")
        return 0

    batch_size = 64
    written_count = 0
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=batch
            )
            written_count += len(batch)
        except Exception as e:
            logger.error("Error upserting batch %d-%d to Qdrant: %s", i, i + len(batch), e)

    logger.info("Successfully upserted %d/%d points into Qdrant.", written_count, len(points))
    return written_count


# ---------------------------------------------------------------------------
# Neo4j Graph Database Ingestion
# ---------------------------------------------------------------------------
def write_to_neo4j(
    driver: Driver,
    inventory_records: List[Dict[str, Any]],
    structure_records: List[Dict[str, Any]]
) -> Tuple[int, int]:
    """Writes documents, sections, domains, scenarios, and relationships to Neo4j."""
    logger.info("Writing inventory and structure topology to Neo4j...")
    
    nodes_merged = 0
    relationships_merged = 0

    with driver.session() as session:
        # 1. Ingest Inventory Documents and Domain nodes
        for doc in inventory_records:
            doc_id = doc.get("doc_id", "").strip()
            if not doc_id:
                continue

            doc_type_raw = doc.get("doc_type", "").strip()
            # Normalize label: British spelling 'Judgement' preserved as standard label
            if doc_type_raw.lower() in ("judgement", "judgment"):
                node_label = "Judgement"
            else:
                node_label = "Act"

            domain = doc.get("domain", "").strip()

            doc_query = f"""
            MERGE (d:{node_label} {{doc_id: $doc_id}})
            SET d.title = $title,
                d.domain = $domain,
                d.doc_type = $doc_type,
                d.year = $year,
                d.act_version = $act_version,
                d.court = $court,
                d.judgement_status = $judgement_status,
                d.source_url = $source_url,
                d.status = $status,
                d.pages = $pages,
                d.reviwed_by = $reviwed_by,
                d.notes = $notes
            """
            params = {
                "doc_id": doc_id,
                "title": doc.get("title", "").strip(),
                "domain": domain,
                "doc_type": doc_type_raw,
                "year": doc.get("year", "").strip(),
                "act_version": doc.get("act_version", "").strip(),
                "court": doc.get("court", "").strip(),
                "judgement_status": doc.get("judgement_status", "").strip(),
                "source_url": doc.get("source_url", "").strip(),
                "status": doc.get("status", "").strip(),
                "pages": doc.get("pages", "").strip(),
                "reviwed_by": doc.get("reviwed_by", doc.get("reviewed_by", "")).strip(),
                "notes": doc.get("notes", "").strip(),
            }

            try:
                session.run(doc_query, params)
                nodes_merged += 1

                if domain:
                    domain_query = """
                    MERGE (dom:Domain {name: $domain})
                    WITH dom
                    MATCH (d {doc_id: $doc_id})
                    MERGE (d)-[:BELONGS_TO_DOMAIN]->(dom)
                    """
                    session.run(domain_query, {"domain": domain, "doc_id": doc_id})
                    nodes_merged += 1
                    relationships_merged += 1

            except Exception as e:
                logger.error("Neo4j error on document %s: %s", doc_id, e)

        # 2. Ingest Structure (Sections, Scenarios, and Related Docs)
        for row in structure_records:
            doc_id = row.get("doc_id", "").strip()
            sec_num = row.get("section_number", "").strip()
            sec_title = row.get("section_title", "").strip()

            if not sec_num and not sec_title:
                continue

            section_uid = f"{doc_id}::{sec_num or sec_title}"
            relevance_tags = parse_delimited_list(row.get("relevance_tags", ""))
            business_scenario = row.get("business_scenario", "").strip()
            related_docs = parse_delimited_list(row.get("related_doc_ids", ""))

            section_query = """
            MERGE (s:Section {uid: $section_uid})
            SET s.doc_id = $doc_id,
                s.section_number = $section_number,
                s.section_title = $section_title,
                s.chapter_number = $chapter_number,
                s.chapter_title = $chapter_title,
                s.page_number = $page_number,
                s.unit_type = $unit_type,
                s.business_scenario = $business_scenario,
                s.relevance_tags = $relevance_tags,
                s.business_relevance = $business_relevance,
                s.plain_summary = $plain_summary
            WITH s
            MATCH (d {doc_id: $doc_id})
            MERGE (d)-[:HAS_SECTION]->(s)
            """
            section_params = {
                "section_uid": section_uid,
                "doc_id": doc_id,
                "section_number": sec_num,
                "section_title": sec_title,
                "chapter_number": row.get("chapter_number", "").strip(),
                "chapter_title": row.get("chapter_title", "").strip(),
                "page_number": row.get("page_number", "").strip(),
                "unit_type": row.get("unit_type", "").strip(),
                "business_scenario": business_scenario,
                "relevance_tags": relevance_tags,
                "business_relevance": row.get("business_relevance", "").strip(),
                "plain_summary": row.get("plain_summary", "").strip(),
            }

            try:
                session.run(section_query, section_params)
                nodes_merged += 1
                relationships_merged += 1

                # Scenario Node & Relationship
                if business_scenario:
                    scenario_query = """
                    MATCH (s:Section {uid: $section_uid})
                    MERGE (bs:BusinessScenario {name: $scenario})
                    MERGE (s)-[:RELATES_TO_SCENARIO]->(bs)
                    """
                    session.run(scenario_query, {"section_uid": section_uid, "scenario": business_scenario})
                    nodes_merged += 1
                    relationships_merged += 1

                # Cross-document RELATED_TO Links
                for rel_doc_id in related_docs:
                    rel_query = """
                    MATCH (s:Section {uid: $section_uid})
                    MATCH (target {doc_id: $rel_doc_id})
                    MERGE (s)-[:RELATED_TO]->(target)
                    """
                    session.run(rel_query, {"section_uid": section_uid, "rel_doc_id": rel_doc_id})
                    relationships_merged += 1

            except Exception as e:
                logger.error("Neo4j error on structure row %s (%s): %s", doc_id, section_uid, e)

    return nodes_merged, relationships_merged


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    load_dotenv()

    # Validate Environment Variables
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    creds_path = os.getenv("GOOGLE_CREDENTIALS_PATH")
    sheet_id = os.getenv("SHEET_ID")
    qdrant_url = os.getenv("QDRANT_URL", os.getenv("QDRANT_HOST", "http://localhost:6333"))

    if not all([neo4j_password, creds_path, sheet_id]):
        logger.error("Missing required environment variables. Please check your .env file.")
        sys.exit(1)

    if not os.path.exists(creds_path):
        logger.error("Google service account credentials file not found at: %s", creds_path)
        sys.exit(1)

    # 1. Connect to External Services
    logger.info("Initializing API and Database connections...")
    try:
        gc = gspread.service_account(filename=creds_path)
        qdrant_client = QdrantClient(url=qdrant_url)
        neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        neo4j_driver.verify_connectivity()
    except Exception as e:
        logger.error("Initialization failure: %s", e)
        sys.exit(1)

    # 2. Fetch Google Sheets Data
    try:
        inventory_records = load_inventory(gc, sheet_id)
        structure_records = load_structure(gc, sheet_id)
        tag_vocab = load_tag_vocabulary(gc, sheet_id)
        scenario_vocab = load_business_scenarios(gc, sheet_id)
    except Exception as e:
        logger.error("Failed to read required data from Google Sheets: %s", e)
        sys.exit(1)

    # Inventory lookup mapping
    inventory_lookup: Dict[str, Dict[str, Any]] = {
        row.get("doc_id", "").strip(): row for row in inventory_records if row.get("doc_id", "").strip()
    }

    # Group structure rows by document for contextual progress logging
    sections_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in structure_records:
        sections_by_doc[row.get("doc_id", "").strip()].append(row)

    # 3. Process, Validate, and Prepare Chunks
    logger.info("--- Starting Chunk Preparation and Soft Validation ---")
    
    chunks_to_embed = []
    chunk_metadata = []
    skipped_rows = []
    unknown_tags = set()
    unknown_scenarios = set()

    for doc_id, doc_meta in inventory_lookup.items():
        doc_title = doc_meta.get("title", "Untitled")
        doc_sections = sections_by_doc.get(doc_id, [])
        logger.info("Processing %s: %s — %d section(s) found", doc_id, doc_title, len(doc_sections))

        for row in doc_sections:
            row_num = row.get("_row_number", "Unknown")
            sec_num = row.get("section_number", "").strip()
            sec_title = row.get("section_title", "").strip()
            chunk_text = build_chunk_text(row, inventory_lookup)

            if not chunk_text:
                reason = "Both section_number and section_title are empty."
                skipped_rows.append((doc_id, row_num, reason))
                logger.warning("Skipping Row %s in %s: %s", row_num, doc_id, reason)
                continue

            # Soft validation: Relevance Tags
            row_tags = parse_delimited_list(row.get("relevance_tags", ""))
            if tag_vocab:
                for tag in row_tags:
                    if tag not in tag_vocab:
                        unknown_tags.add(tag)
                        logger.warning("Soft warning: Unknown tag '%s' in %s (Row %s)", tag, doc_id, row_num)

            # Soft validation: Business Scenario
            scenario = row.get("business_scenario", "").strip()
            if scenario and scenario_vocab:
                if scenario not in scenario_vocab:
                    unknown_scenarios.add(scenario)
                    logger.warning("Soft warning: Unknown scenario '%s' in %s (Row %s)", scenario, doc_id, row_num)

            point_id = get_deterministic_uuid(doc_id, sec_num, sec_title)
            payload = {
                "doc_id": doc_id,
                "domain": doc_meta.get("domain", "").strip(),
                "doc_type": doc_meta.get("doc_type", "").strip(),
                "unit_type": row.get("unit_type", "").strip(),
                "section_number": sec_num,
                "section_title": sec_title,
                "relevance_tags": row_tags,
                "business_scenario": scenario,
                "chunk_text": chunk_text,
                "title": doc_title
            }

            chunks_to_embed.append(chunk_text)
            chunk_metadata.append({
                "point_id": point_id,
                "payload": payload
            })

    # 4. Generate Embeddings locally
    logger.info("--- Embedding Generation ---")
    if chunks_to_embed:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        vectors = embed_chunks(chunks_to_embed, model)
        for meta, vec in zip(chunk_metadata, vectors):
            meta["vector"] = vec
    else:
        logger.warning("No valid text chunks found to embed.")

    # 5. Write to Vector DB (Qdrant)
    logger.info("--- Writing to Qdrant ---")
    qdrant_written = 0
    try:
        qdrant_written = write_to_qdrant(qdrant_client, chunk_metadata)
    except Exception as e:
        logger.error("Failed writing to Qdrant: %s", e)

    # 6. Write to Graph DB (Neo4j)
    logger.info("--- Writing to Neo4j ---")
    neo4j_nodes = 0
    neo4j_rels = 0
    try:
        neo4j_nodes, neo4j_rels = write_to_neo4j(neo4j_driver, inventory_records, structure_records)
    except Exception as e:
        logger.error("Failed writing to Neo4j: %s", e)
    finally:
        neo4j_driver.close()

    # ---------------------------------------------------------------------------
    # Summary Report
    # ---------------------------------------------------------------------------
    logger.info("==========================================================")
    logger.info("                 INGESTION SUMMARY REPORT                 ")
    logger.info("==========================================================")
    logger.info("Documents Inventory Processed     : %d", len(inventory_lookup))
    logger.info("Structure Rows Read               : %d", len(structure_records))
    logger.info("Chunks Successfully Embedded      : %d", len(chunks_to_embed))
    logger.info("Qdrant Points Written             : %d", qdrant_written)
    logger.info("Neo4j Operations (Nodes / Rels)   : %d / %d", neo4j_nodes, neo4j_rels)
    logger.info("Rows Skipped                      : %d", len(skipped_rows))
    
    if skipped_rows:
        logger.info("--- Skipped Rows Breakdown ---")
        for doc_id, row_idx, reason in skipped_rows:
            logger.info("  * Doc: %s, Row: %s — %s", doc_id, row_idx, reason)

    if unknown_tags:
        logger.info("--- Unknown Tags Encountered (%d) ---", len(unknown_tags))
        logger.info("  %s", ", ".join(sorted(unknown_tags)))

    if unknown_scenarios:
        logger.info("--- Unknown Scenarios Encountered (%d) ---", len(unknown_scenarios))
        logger.info("  %s", ", ".join(sorted(unknown_scenarios)))

    logger.info("==========================================================")
    logger.info("Pipeline run completed successfully.")


if __name__ == "__main__":
    main()