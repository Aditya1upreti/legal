"""
Legal RAG — Retrieval + Generation Layer
Business Legal Assistant
======================================

 Takes a user's question and produces a grounded, cited answer using:
  - Qdrant vector similarity search (all-MiniLM-L6-v2)
  - Neo4j graph expansion (Act/Judgement → Section → related docs)
  - Gemini generation via Google AI API
"""
import os
import re
import sys
import logging
from typing import Dict, List, Any, Optional, Tuple

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import google.genai as genai
from neo4j import GraphDatabase


# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LegalRAG")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = "legal_chunks"
VECTOR_DIMENSION = 384
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Helper: Embedding
# ---------------------------------------------------------------------------
def get_embedding_model() -> SentenceTransformer:
    """Load and return the local SentenceTransformer model."""
    logger.info("Loading embedding model '%s'...", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def embed_text(model: SentenceTransformer, text: str) -> List[float]:
    """Encode a single text string into a dense vector."""
    vec = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return vec.tolist()

# ---------------------------------------------------------------------------
# Function 1: retrieve_context
# ---------------------------------------------------------------------------
def retrieve_context(question: str, top_k: int = 5) -> dict:
    """
    Step A — Vector search against Qdrant.
    Step B — Graph expansion against Neo4j.
    Step C — Return structured dict.
    """
    # Load env
    load_dotenv()
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    if not neo4j_password:
        logger.error("Missing NEO4J_PASSWORD in environment.")
        return {"vector_hits": [], "graph_context": []}

    # Connect to Qdrant
    qdrant_client = QdrantClient(url=qdrant_url)

    # Step A — Vector search
    model = get_embedding_model()
    question_embedding = embed_text(model, question)

    try:
        qdrant_hits = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=question_embedding,
            limit=top_k
        )
    except Exception as e:
        logger.error("Qdrant vector search failed: %s", e)
        qdrant_hits = []

    # Collect doc_id and section_number from each hit's payload
    vector_hits = []
    doc_section_pairs = set()  # (doc_id, section_number) for deduplication
    for hit in qdrant_hits:
        payload = hit.payload if hasattr(hit, 'payload') else {}
        doc_id = payload.get("doc_id", "")
        section_number = payload.get("section_number", "")
        score = getattr(hit, 'score', None)
        vector_hits.append({
            "doc_id": doc_id,
            "section_number": section_number,
            "payload": payload,
            "score": score
        })
        if doc_id and section_number:
            doc_section_pairs.add((doc_id, section_number))

    # Step B — Graph expansion against Neo4j
    graph_context = []
    seen_section_uids = set()  # deduplicate sections

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as session:
            for doc_id, section_number in doc_section_pairs:
                # a) Parent Act/Judgement metadata via Section match
                parent_query = """
                MATCH (s:Section {doc_id: $doc_id, section_number: $section_number})
                  MATCH (parent)-[:HAS_SECTION]->(s)
                RETURN parent.doc_id AS parent_doc_id,
                   parent.title AS parent_title,
                   parent.year AS parent_year,
                   parent.court AS parent_court,
                   parent.judgement_status AS parent_judgement_status,
                   parent.source_url AS parent_source_url,
                   parent.doc_type AS parent_doc_type
                """
                try:
                    parent_result = session.run(parent_query, doc_id=doc_id, section_number=section_number).single()
                    if parent_result:
                        parent_info = {
                            "title": parent_result["parent_title"],
                            "year": parent_result["parent_year"],
                            "court": parent_result["parent_court"],
                            "judgement_status": parent_result["parent_judgement_status"],
                            "source_url": parent_result["parent_source_url"],
                            "doc_type": parent_result["parent_doc_type"],
                            "why": "direct match"
                        }
                        graph_context.append(parent_info)
                except Exception as e:
                    logger.error("Neo4j parent query error for %s:%s: %s", doc_id, section_number, e)

                # b) Other sections connected via (Section)-[:RELATED_TO]->(otherDoc)
                try:
                    related_query = """
                    MATCH (s:Section {doc_id: $doc_id, section_number: $section_number})
                      MATCH (s)-[:RELATED_TO]->(related)
                    RETURN related.doc_id AS related_doc_id,
                           related.title AS related_title,
                           related.doc_type AS related_doc_type,
                           related.section_number AS related_section_number,
                           related.section_title AS related_section_title,
                           related.business_scenario AS related_business_scenario
                    """
                    related_result = session.run(related_query, doc_id=doc_id, section_number=section_number)
                    for record in related_result:
                        related_uid = f"{record['related_doc_id']}::{record['related_section_number']}"
                        if related_uid not in seen_section_uids:
                            seen_section_uids.add(related_uid)
                            graph_context.append({
                                "title": record["related_title"],
                                "section_number": record["related_section_number"],
                                "doc_type": record["related_doc_type"],
                                "business_scenario": record["related_business_scenario"],
                                "why": "related via RELATED_TO",
                                "related_doc_id": record["related_doc_id"]
                            })
                except Exception as e:
                    logger.error("Neo4j related-to query error for %s:%s: %s", doc_id, section_number, e)

                # c) Sibling sections under the same parent Act (same doc_id)
                #    that share the same business_scenario as the matched section
                try:
                    sibling_query = """
                    MATCH (s:Section {doc_id: $doc_id, section_number: $section_number})
                      MATCH (s)-[:RELATES_TO_SCENARIO]->(bs:BusinessScenario)
                      MATCH (other:Section)
                      WHERE other.doc_id = $doc_id
                        AND other.section_number <> $section_number
                        AND other.business_scenario = bs.name
                      RETURN other.section_number AS section_number,
                             other.section_title AS section_title,
                             other.business_scenario AS business_scenario,
                             other.plain_summary AS plain_summary,
                             other.business_relevance AS business_relevance
                    """
                    sibling_result = session.run(sibling_query, doc_id=doc_id, section_number=section_number)
                    for record in sibling_result:
                        sibling_uid = f"{doc_id}::{record['section_number']}"
                        if sibling_uid not in seen_section_uids:
                            seen_section_uids.add(sibling_uid)
                            graph_context.append({
                                "title": doc_id,  # will be refined
                                "section_number": record["section_number"],
                                "business_scenario": record["business_scenario"],
                                "plain_summary": record["plain_summary"],
                                "business_relevance": record["business_relevance"],
                                "why": "sibling section, same business_scenario"
                            })
                except Exception as e:
                    logger.error("Neo4j sibling query error for %s:%s: %s", doc_id, section_number, e)

        driver.close()

    except Exception as e:
        logger.error("Neo4j connection/graph expansion failed: %s", e)

    # Step C — Return structured dict
    result = {
        "vector_hits": vector_hits,
        "graph_context": graph_context
    }
    return result


# ---------------------------------------------------------------------------
# Function 2: build_prompt
# ---------------------------------------------------------------------------
def build_prompt(question: str, context: dict) -> str:
    """
    Format the retrieved context into a clean, labeled block for Claude.
    Clearly separate direct matches from graph-expanded related context.
    """
    vector_hits = context.get("vector_hits", [])
    graph_context = context.get("graph_context", [])

    # Build source blocks
    source_blocks = []

    # Direct vector hits
    if vector_hits:
        direct_lines = ["[Direct Vector Matches]"]
        for i, hit in enumerate(vector_hits, 1):
            payload = hit.get("payload", {})
            doc_id = payload.get("doc_id", "")
            sec_num = payload.get("section_number", "")
            title = payload.get("title", "")
            domain = payload.get("domain", "")
            relevance_tags = payload.get("relevance_tags", [])
            business_scenario = payload.get("business_scenario", "")
            chunk_text = payload.get("chunk_text", "")

            # Build source label
            source_label = f"[Source {i} — {title}, Section {sec_num}, {domain}]"
            direct_lines.append(source_label)

            # Add the chunk text / business relevance + plain summary
            body_parts = []
            if business_scenario:
                body_parts.append(f"Business Scenario: {business_scenario}")
            if chunk_text:
                body_parts.append(chunk_text)
            # Also include plain_summary if available in payload
            plain_summary = payload.get("plain_summary", "")
            if plain_summary and plain_summary not in chunk_text:
                body_parts.append(f"Summary: {plain_summary}")

            body_text = " ".join(body_parts)
            # Truncate if very long for prompt clarity
            if len(body_text) > 800:
                body_text = body_text[:800] + "..."

            direct_lines.append(body_text)
            direct_lines.append("")  # blank line between sources

        source_blocks.append("\n".join(direct_lines))

    # Graph-expanded context
    if graph_context:
        graph_lines = ["[Graph-Expanded Context]"]
        seen_whys = {}
        for i, ctx in enumerate(graph_context, 1):
            why = ctx.get("why", "context")
            title = ctx.get("title", "Unknown")
            sec_num = ctx.get("section_number", "")
            business_relevance = ctx.get("business_relevance", "")
            plain_summary = ctx.get("plain_summary", "")

            # Deduplicate by why category
            if why not in seen_whys:
                seen_whys[why] = True
                label = f"[Graph Source {i} — {title}, Section {sec_num}] ({why})"
                graph_lines.append(label)

                body_parts = []
                if business_relevance:
                    body_parts.append(f"Relevance: {business_relevance}")
                if plain_summary:
                    body_parts.append(f"Summary: {plain_summary}")

                body_text = " ".join(body_parts)
                if len(body_text) > 600:
                    body_text = body_text[:600] + "..."

                graph_lines.append(body_text)
                graph_lines.append("")

        source_blocks.append("\n".join(graph_lines))

    # Combine all blocks
    all_blocks = "\n".join(source_blocks) if source_blocks else "[No retrieved context found.]"

    # System instruction
    system_instruction = (
        "You are a legal research assistant for the Business Legal Assistant system. "
        "Answer the user's question using ONLY the provided source material below. "
        "If the sources do not contain enough information to answer the question, "
        "say explicitly: \"The provided sources do not contain enough information to answer this question reliably.\" "
        "Do not guess or use external knowledge. "
        "For every claim you make, cite the specific Act/Section or Judgment used, "
        "using the format: (Source: <title>, <section_number>). "
        "If multiple sources support a point, list them all: (Source: <title1>, <sec1>) and (Source: <title2>, <sec2>)."
    )

    prompt = f"""{system_instruction}

--- Retrieved Legal Sources ---

{all_blocks}

--- User Question ---

{question}

--- Instructions ---

Provide a grounded, cited answer based solely on the sources above"""

    return prompt


# ---------------------------------------------------------------------------
# Function 3: generate_answer
# ---------------------------------------------------------------------------
def generate_answer(question: str, context: dict) -> str:
    """
    Call the Google Gemini API with the prompt built from context.
    Handle API errors gracefully, including rate limits.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY not found in environment variables.\nSet GEMINI_API_KEY in your .env file."

    try:
        client = genai.Client(api_key=api_key)

        prompt = build_prompt(question, context)

        # First attempt
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )
        except Exception as e:
            # Check for rate limit error (429) and retry once
            err_str = str(e)
            if "429" in err_str or "rate limit" in err_str.lower() or "quota" in err_str.lower():
                logger.info("Rate limit hit, retrying after brief wait...")
                import time
                time.sleep(2)
                # Retry once
                response = client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
            else:
                raise

        return response.text

    except Exception as e:
        return f"Error: Gemini API request failed: {str(e)}"


# ---------------------------------------------------------------------------
# Function 4: answer_question
# ---------------------------------------------------------------------------
def answer_question(question: str) -> dict:
    """
    Orchestrate: retrieve_context() -> build_prompt() -> generate_answer()
    Returns: {"question": ..., "answer": ..., "sources_used": [...], "raw_context": ..., "citation_check": ...}
    """
    raw_context = retrieve_context(question, top_k=5)

    # Build sources_used list from both vector_hits and graph_context
    sources_used = set()
    sources_list = []  # list of (title, sec_num, source_url) for easy checking

    # From vector hits
    for hit in raw_context.get("vector_hits", []):
        payload = hit.get("payload", {})
        doc_id = payload.get("doc_id", "")
        sec_num = payload.get("section_number", "")
        title = payload.get("title", "")
        source_url = payload.get("source_url", "")
        if doc_id and sec_num:
            sources_used.add((title, sec_num, source_url))
            sources_list.append((title.strip().lower(), sec_num.strip(), source_url))

    # From graph context
    for ctx in raw_context.get("graph_context", []):
        title = ctx.get("title", "Unknown")
        sec_num = ctx.get("section_number", "")
        # Try to find source_url from the original payload
        # We'll use a placeholder if not available
        if sec_num and title != "Unknown":
            sources_used.add((title, sec_num, ""))
            sources_list.append((title.strip().lower(), sec_num.strip(), ""))

    answer = generate_answer(question, raw_context)

    # --- Citation validation ---
    verified = []
    unverified = []
    # Extract citations of the form (Source: <title>, <section>) from the answer
    import re
    citation_pattern = re.compile(r"\(Source:\s*([^\),]+)\s*,\s*([^)]+)\)", re.IGNORECASE)
    found_citations = citation_pattern.findall(answer)
    
    # Build a set of available (title_lower, sec_num) for quick lookup
    available = {(t, s) for t, s, _ in sources_list}
    
    for title, sec in found_citations:
        title_clean = title.strip().lower()
        sec_clean = sec.strip()
        if (title_clean, sec_clean) in available or (title_clean, sec_clean.rstrip('.')) in available:
            verified.append((title, sec))
        else:
            unverified.append((title, sec))
    
    citation_check = {
        "verified": verified,
        "unverified": unverified,
        "total": len(found_citations)
    }

    return {
        "question": question,
        "answer": answer,
        "sources_used": sorted(list(sources_used)),
        "raw_context": raw_context,
        "citation_check": citation_check
    }


# ---------------------------------------------------------------------------
# CLI / Test Mode
# ---------------------------------------------------------------------------
def pretty_print_result(result: dict):
    """Pretty-print the answer result for CLI display."""
    question = result["question"]
    answer = result["answer"]
    sources = result["sources_used"]

    print("\n" + "=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    print("\n--- Retrieved Sources ---")
    # Group sources by type
    direct_sources = [(t, s) for t, s, _ in sources if _ == ""]  # no source_url marker
    # Actually let's just print all sources with their info

    # Re-display from raw context
    raw = result["raw_context"]
    vector_hits = raw.get("vector_hits", [])
    graph_context = raw.get("graph_context", [])

    print("\n[Direct Vector Matches]")
    if vector_hits:
        for i, hit in enumerate(vector_hits, 1):
            payload = hit.get("payload", {})
            title = payload.get("title", "")
            sec_num = payload.get("section_number", "")
            domain = payload.get("domain", "")
            score = hit.get("score", None)
            score_str = f"{score:.4f}" if score is not None else "N/A (graph-expanded)"
            print(f"  {i}. {title}, Section {sec_num} ({domain}) — score: {score_str}")
    else:
        print("  None found.")

    print("\n[Graph-Expanded Context]")
    if graph_context:
        # Group by why
        by_why = {}
        for ctx in graph_context:
            why = ctx.get("why", "context")
            if why not in by_why:
                by_why[why] = []
            by_why[why].append(ctx)

        for why, ctxs in by_why.items():
            print(f"  ({why}):")
            for ctx in ctxs:
                title = ctx.get("title", "Unknown")
                sec_num = ctx.get("section_number", "")
                print(f"    - {title}, Section {sec_num}")
    else:
        print("  None found.")

    print("\n--- FINAL ANSWER ---")
    print(answer)
    print("=" * 70)

    # Print sources summary
    print("\n--- Sources Referenced ---")
    for title, sec_num, source_url in sources:
        url_part = f", URL: {source_url}" if source_url else ""
        print(f"  - {title}, Section {sec_num}{url_part}")


def run_cli():
    """Run the script in interactive CLI mode."""
    print("Business Legal Assistant — Retrieval + Generation")
    print("=" * 50)
    try:
        question = input("\nEnter your legal question: ").strip()
    except EOFError:
        print("\nNo input available. Use --batch mode or run interactively.")
        return
    if not question:
        print("No question entered. Exiting.")
        return

    result = answer_question(question)
    pretty_print_result(result)


def run_batch(batch_file: str):
    """Run the pipeline in batch mode, reading questions from a file."""
    if not os.path.exists(batch_file):
        logger.error("Batch file not found: %s", batch_file)
        print(f"Error: Batch file '{batch_file}' not found.")
        return

    with open(batch_file, "r", encoding="utf-8") as f:
        questions = [line.strip() for line in f if line.strip()]

    if not questions:
        print("No questions found in the batch file.")
        return

    print(f"\nRunning batch mode with {len(questions)} question(s).")
    print("=" * 70)

    for idx, question in enumerate(questions, 1):
        print(f"\n[{idx}/{len(questions)}] Processing: {question}")
        print("-" * 70)
        result = answer_question(question)
        pretty_print_result(result)

    print("\n" + "=" * 70)
    print("Batch processing complete.")
    print("=" * 70)


def main():
    """Entry point."""
    load_dotenv()

    # Check for --batch flag
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_file = os.getenv("BATCH_QUESTIONS_FILE", "scripts/test_questions.txt")
        # Allow overriding the batch file path
        if len(sys.argv) > 2:
            batch_file = sys.argv[2]
        run_batch(batch_file)
    else:
        run_cli()


if __name__ == "__main__":
    main()