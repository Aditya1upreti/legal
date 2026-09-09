"""
Evaluation Script: GraphRAG vs. Vector-Only Retrieval
=====================================================

Demonstrates the empirical advantage of GraphRAG (vector search + Neo4j graph expansion)
over plain vector search alone. Built to accompany the Business Legal Assistant project.

Run with:
    python scripts/evaluate_graphrag.py
"""

import os
import sys
import logging
import json
import matplotlib.pyplot
from typing import Dict, List, Any, Tuple, Set

# Add project root to path so we can import the pipeline script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.retrieve_and_answer import retrieve_context, answer_question

# ---------------------------------------------------------------------------
# Logging Configuration (same style as the rest of the codebase)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GraphRAGEvaluation")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), "test_questions.txt")
CHART_FILE = os.path.join(os.path.dirname(__file__), "evaluation_chart.png")

# ---------------------------------------------------------------------------
# Helper: Extract the "why" label for a graph_context entry
# ---------------------------------------------------------------------------
def get_why_label(graph_context: List[Dict[str, Any]], target_uid: str) -> str:
    """Find the 'why' label for a section uid in graph_context."""
    for entry in graph_context:
        # UIDs in graph_context were built as f"{doc_id}::{section_number}" internally
        # but the entries may not have a uid field — we match by available fields.
        why = entry.get("why", "")
        if why:
            return why
    return "unknown"


# ---------------------------------------------------------------------------
# FUNCTION: retrieve_vector_only
# ---------------------------------------------------------------------------
def retrieve_vector_only(question: str, top_k: int = 5) -> dict:
    """
    Pure vector-search baseline — same Qdrant step as retrieve_context()
    but WITHOUT the Neo4j graph expansion.
    Returns the vector_hits list from retrieve_context's internal logic.
    """
    logger.info("Retrieving vector-only context for: %s", question[:60])

    # Load env + Qdrant setup (reused from retrieve_and_answer.py patterns)
    from dotenv import load_dotenv
    load_dotenv()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")

    if not neo4j_password:
        logger.error("Missing NEO4J_PASSWORD in environment.")
        return {"vector_hits": [], "graph_context": [], "score": None}

    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient

    # Connect to Qdrant
    qdrant_client = QdrantClient(url=qdrant_url)

    # Embed the question using the same local model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    question_embedding = model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    try:
        qdrant_hits = qdrant_client.query_points(
            collection_name="legal_chunks",
            query=question_embedding.tolist(),
            limit=top_k
        )
    except Exception as e:
        logger.error("Qdrant vector search failed: %s", e)
        qdrant_hits = []

    # Collect hits with payload + score (mirrors retrieve_context step A)
    vector_hits = []
    for hit in qdrant_hits:
        payload = hit.payload if hasattr(hit, 'payload') else {}
        score = getattr(hit, 'score', None)
        vector_hits.append({
            "doc_id": payload.get("doc_id", ""),
            "section_number": payload.get("section_number", ""),
            "title": payload.get("title", ""),
            "domain": payload.get("domain", ""),
            "score": score
        })

    logger.info("Vector-only returned %d hits.", len(vector_hits))
    return {"vector_hits": vector_hits, "graph_context": [], "score": None}


# ---------------------------------------------------------------------------
# FUNCTION: retrieve_graphrag
# ---------------------------------------------------------------------------
def retrieve_graphrag(question: str, top_k: int = 5) -> dict:
    """
    Calls the existing retrieve_context() from retrieve_and_answer.py
    — vector search + graph expansion combined.
    """
    logger.info("Retrieving GraphRAG context for: %s", question[:60])
    result = retrieve_context(question, top_k=top_k)
    logger.info("GraphRAG returned %d vector_hits + %d graph_context items.",
                len(result.get("vector_hits", [])), len(result.get("graph_context", [])))
    return result


# ---------------------------------------------------------------------------
# FUNCTION: compare_retrieval
# ---------------------------------------------------------------------------
def compare_retrieval(question: str) -> dict:
    """
    Runs both retrieve_vector_only() and retrieve_graphrag() for the same question.
    Compares results and returns a dict with comparison stats.
    """
    logger.info("Comparing retrieval methods for question: %s", question[:60])

    # --- Vector-only ---
    try:
        vec_result = retrieve_vector_only(question, top_k=DEFAULT_TOP_K)
    except Exception as e:
        logger.error("Vector-only retrieval failed: %s", e)
        vec_result = {"vector_hits": [], "graph_context": []}

    # --- GraphRAG ---
    try:
        gra_result = retrieve_graphrag(question, top_k=DEFAULT_TOP_K)
    except Exception as e:
        logger.error("GraphRAG retrieval failed: %s", e)
        gra_result = {"vector_hits": [], "graph_context": []}

    # --- Extract sections from each method ---
    vec_sections: Set[Tuple[str, str]] = set()  # (doc_id, section_number)
    for hit in vec_result.get("vector_hits", []):
        doc_id = hit.get("doc_id", "")
        sec_num = hit.get("section_number", "")
        if doc_id and sec_num:
            vec_sections.add((doc_id, sec_num))

    gra_sections: Set[Tuple[str, str]] = set()
    gra_why_map: Dict[Tuple[str, str], str] = {}  # (doc_id, sec_num) -> why label

    for hit in gra_result.get("vector_hits", []):
        doc_id = hit.get("doc_id", "")
        sec_num = hit.get("section_number", "")
        if doc_id and sec_num:
            gra_sections.add((doc_id, sec_num))

    # Graph-expanded sections: from graph_context, collect sections with their "why"
    gra_expanded_sections: Set[Tuple[str, str]] = set()
    for entry in gra_result.get("graph_context", []):
        # Build a key from available fields; try title+section_number or doc_id+section_number
        title = entry.get("title", "")
        sec_num = entry.get("section_number", "")
        why = entry.get("why", "unknown")
        if sec_num:
            key = (title, sec_num)
            gra_sections.add(key)
            gra_why_map[key] = why

    # --- Compute differences ---
    # Sections that appear ONLY in GraphRAG (not in vector-only)
    graphrag_only: Set[Tuple[str, str]] = gra_sections - vec_sections

    # Why labels for each GraphRAG-only addition
    only_why_labels: List[str] = []
    for key in graphrag_only:
        why = gra_why_map.get(key, "unknown")
        only_why_labels.append(why)

    # Also collect human-readable descriptions
    only_descriptions: List[str] = []
    for key in graphrag_only:
        title, sec_num = key
        why = gra_why_map.get(key, "unknown")
        only_descriptions.append(f"{title}, Section {sec_num} ({why})")

    # --- Build comparison dict ---
    comparison = {
        "question": question,
        "vector_only_count": len(vec_sections),
        "graphrag_total_count": len(gra_sections),
        "graphrag_only_count": len(graphrag_only),
        "graphrag_only_sections": sorted(list(graphrag_only)),
        "graphrag_only_why_labels": only_why_labels,
        "graphrag_only_descriptions": only_descriptions,
        "vec_hits": vec_result.get("vector_hits", []),
        "gra_hits": gra_result.get("vector_hits", []),
        "gra_context": gra_result.get("graph_context", []),
    }

    logger.info(
        "Comparison: vector=%d, GraphRAG total=%d, GraphRAG-only=%d",
        len(vec_sections), len(gra_sections), len(graphrag_only)
    )
    return comparison


# ---------------------------------------------------------------------------
# FUNCTION: run_full_evaluation
# ---------------------------------------------------------------------------
def run_full_evaluation(questions_file: str = QUESTIONS_FILE) -> None:
    """
    Reads all questions from the test questions file, runs compare_retrieval() for each,
    prints a per-question comparison table, and an aggregate summary.
    Also generates a bar chart saved as scripts/evaluation_chart.png.
    """
    # Read questions
    questions: List[str] = []
    if os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    if not questions:
        logger.warning("No questions found in %s — using hardcoded fallback.", questions_file)
        questions = [
            "Vendor breached a contract — can damages be claimed?",
            "An MSME supplier wasn't paid within 45 days — what can they do?",
            "Can an LLP partner be held personally liable for the firm's debts?",
            "What happens if a company delays allotting shares after receiving investment money?",
            "Is a builder liable to a homebuyer for delayed possession under consumer law?",
        ]

    logger.info("Starting evaluation with %d question(s).", len(questions))

    # Per-question results storage
    per_question_results: List[dict] = []
    total_only = 0  # count of questions with at least one GraphRAG-only addition
    total_questions = len(questions)
    sum_additional = 0  # sum of graphrag_only_count across questions

    # Chart data storage
    vec_counts: List[int] = []
    gra_counts: List[int] = []
    only_counts: List[int] = []

    for idx, question in enumerate(questions, 1):
        try:
            comp = compare_retrieval(question)
        except Exception as e:
            logger.error("Error evaluating question %d (%s): %s", idx, question[:50], e)
            comp = {
                "question": question,
                "vector_only_count": 0,
                "graphrag_total_count": 0,
                "graphrag_only_count": 0,
                "graphrag_only_sections": [],
                "graphrag_only_why_labels": [],
                "graphrag_only_descriptions": [],
            }

        per_question_results.append(comp)

        vec_counts.append(comp.get("vector_only_count", 0))
        gra_counts.append(comp.get("graphrag_total_count", 0))
        only_counts.append(comp.get("graphrag_only_count", 0))

        only_count = comp.get("graphrag_only_count", 0)
        if only_count > 0:
            total_only += 1
        sum_additional += only_count

        logger.info(
            "Q%d: vector=%d, GraphRAG total=%d, GraphRAG-only=%d — %s",
            idx, comp.get("vector_only_count", 0),
            comp.get("graphrag_total_count", 0),
            only_count,
            comp.get("graphrag_only_descriptions", [])[:1]
        )

    # --- Aggregate summary ---
    avg_additional = sum_additional / total_questions if total_questions > 0 else 0
    questions_with_addition_pct = (total_only / total_questions * 100) if total_questions > 0 else 0

    logger.info("=" * 60)
    logger.info("AGGREGATE SUMMARY across %d questions:", total_questions)
    logger.info("  Average additional relevant sections surfaced by GraphRAG per question: %.2f", avg_additional)
    logger.info("  Percentage of questions with at least one GraphRAG-only addition: %.1f%%", questions_with_addition_pct)
    logger.info("=" * 60)

    # Print per-question table (human-readable)
    print("\n" + "=" * 80)
    print("GRAPHRAG vs. VECTOR-ONLY RETRIEVAL — PER-QUESTION COMPARISON")
    print("=" * 80)

    header_fmt = f"{{:<{25}}} {{:<{15}}} {{:<{15}}} {{:<{15}}} {{:<{40}}}"
    print(header_fmt("Question", "Vector Count", "GraphRAG Total", "GraphRAG-only", "GraphRAG-only Additions"))
    print("-" * 80)

    for i, comp in enumerate(per_question_results, 1):
        q = comp.get("question", "")[:40]
        v = comp.get("vector_only_count", 0)
        g = comp.get("graphrag_total_count", 0)
        o = comp.get("graphrag_only_count", 0)
        descs = "; ".join(comp.get("graphrag_only_descriptions", [])[:2])  # first 2 descs
        print(header_fmt(q, str(v), str(g), str(o), descs))

    print("-" * 80)
    print(
        f"{'AGGREGATE':<{25}} {'':<{15}} {'':<{15}} "
        f"{str(round(avg_additional, 2)):<{15}} {'% questions with GraphRAG benefit: ' + str(round(questions_with_addition_pct, 1)) + '%':<{40}}"
    )
    print("=" * 80 + "\n")

    # --- Generate bar chart ---
    try:
        matplotlib.pyplot.figure(figsize=(10, 6))
        x_labels = [f"Q{i}" for i in range(1, len(questions) + 1)]

        # Bar positions
        x = range(len(questions))

        # Plot vector-only and GraphRAG total as grouped bars
        matplotlib.pyplot.bar(
            [i - 0.2 for i in x],
            vec_counts,
            width=0.4,
            label="Vector-only",
            color="#4C72B0"
        )
        matplotlib.pyplot.bar(
            [i + 0.2 for i in x],
            gra_counts,
            width=0.4,
            label="GraphRAG total",
            color="#55A868"
        )

        matplotlib.pyplot.xlabel("Question Index")
        matplotlib.pyplot.ylabel("Number of Distinct Sections Returned")
        matplotlib.pyplot.title("GraphRAG vs. Vector-Only Retrieval Results")
        matplotlib.pyplot.xticks(ticks=x, labels=x_labels, rotation=30, ha="right")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.tight_layout()

        matplotlib.pyplot.savefig(CHART_FILE)
        matplotlib.pyplot.close()
        logger.info("Bar chart saved to %s", CHART_FILE)
    except Exception as e:
        logger.error("Failed to generate bar chart: %s", e)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Running GraphRAG evaluation (CLI mode).")
    try:
        run_full_evaluation()
    except Exception as e:
        logger.critical("Evaluation run failed: %s", e)
        print(f"\nError during evaluation: {e}")