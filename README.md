# Business Legal Assistant (Lexora)

A GraphRAG legal question-answering system for Indian business law (Contract, Employment, Consumer). 
Legal Acts and Judgments are curated in Google Sheets, embedded locally with sentence-transformers, 
stored in Qdrant (vector search) and Neo4j (knowledge graph). A question is answered by vector search 
plus graph expansion, then an LLM (Gemini) generates a cited answer from only the retrieved context. 
A Streamlit app is the frontend.

## Architecture (in brief)

1. **Google Sheets** — 4 tabs supply the data:
   - **Document Inventory** — doc_id, title, domain, doc_type, year, etc.
   - **Document Structure** — section_number, section_title, business_relevance, plain_summary, 
     relevance_tags, business_scenario, related_doc_ids
   - **Tag Vocabulary** — reference tags
   - **Business Scenarios** — reference scenarios
2. **Ingestion** (`scripts/ingest_pipeline.py`) — Reads Sheets, normalizes headers, chunks text, 
   generates SentenceTransformer embeddings (all-MiniLM-L6-v2, 384 dim), writes to Qdrant and Neo4j.
3. **Qdrant** — Vector store for semantic similarity search (cosine distance, collection: `legal_chunks`).
4. **Neo4j** — Knowledge graph storing Document → Section → Domain → BusinessScenario relationships. 
   Enables graph expansion during retrieval.
5. **Retrieval** — Vector search in Qdrant + optional Neo4j graph expansion (traverse relationships).
6. **Gemini** (`google-genai`) — Generates a grounded, cited answer from the retrieved context, enforcing 
   the `(Source: <title>, <section>)` citation format.
7. **Streamlit** (`app.py`) — Frontend chat interface with example questions, citation pills, and retrieval 
   trace expander.

## Setup Steps

1. **Install dependencies:**
   ```powershell
   python -m venv graph_env
   .\graph_env\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   - Copy `.env.example` to `.env`
   - Fill in: `GEMINI_API_KEY`, `NEO4J_PASSWORD`, `SHEET_ID`, `GOOGLE_CREDENTIALS_PATH`
   - Verify `QDRANT_URL` (default `http://localhost:6333`)

3. **Start the services** (see `start_local.ps1`):
   ```powershell
   .\start_local.ps1
   ```
   - This starts Qdrant Docker container and checks connectivity
   - Start Neo4j Desktop manually and open your database

4. **Run the verification checks** (see `verify.py`):
   - `verify.py check 1` — Google Sheets connection
   - `verify.py check 2` — Qdrant connectivity
   - `verify.py check 3` — Neo4j connectivity

5. **Ingest the data:**
   ```powershell
   .\scripts\ingest_pipeline.py
   ```

6. **Start the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

7. **Run the evaluation:**
   ```powershell
   .\scripts\evaluate_graphrag.py
   ```

8. **Test with batch questions:**
   ```powershell
   .\scripts\retrieve_and_answer.py --batch scripts/test_questions.txt
   ```

## Order to Run Things

1. `start_local.ps1` — Start Qdrant, verify, remind about Neo4j
2. Start Neo4j Desktop and start your database
3. `verify.py check 1` — Google Sheets connection
4. `verify.py check 2` — Qdrant connectivity
5. `verify.py check 3` — Neo4j connectivity
6. `ingest_pipeline.py` — Load Sheets data into Qdrant + Neo4j
7. `retrieve_and_answer.py` — Verify end-to-end QA
8. `streamlit run app.py` — Start the frontend UI
9. `evaluate_graphrag.py` — Run GraphRAG vs vector-only comparison

## Google Sheet Tab Structure

| Tab Name | Key Columns |
|---|---|
| **Document Inventory** | doc_id, title, domain, doc_type, year, act_version, court, judgement_status, source_url, status, pages, reviewed_by (or review\_by) |
| **Document Structure** | doc_id, section_number, section_title, chapter_number, chapter_title, page_number, unit_type, business_scenario, relevance_tags (comma-separated), business_relevance, plain_summary, related_doc_ids |
| **Tag Vocabulary** | tag (one per row — reference set) |
| **Business Scenarios** | scenario (one per row — reference set) |

> **Note:** The loader accepts either `reviewed_by` or `reviwed_by` (misspelled) as the column header — 
> normalize_header() will normalize both to the same key, and the Neo4j write function tries both spellings.

## Troubleshooting

- **"Connection refused" errors** — Usually means Neo4j or Qdrant is not started. 
  Run `start_local.ps1` to start Qdrant, and ensure Neo4j Desktop is running with a database started.
- **Qdnot found error** — Verify the Qdrant container is running: `docker ps | grep qdrant_bla`
- **Neo4j auth error** — Check `NEO4J_USER` and `NEO4J_PASSWORD` in `.env`
- **Sheets API error** — Check `GOOGLE_CREDENTIALS_PATH` and `SHEET_ID` in `.env`
- **Gemini API error** — Check `GEMINI_API_KEY` in `.env`
- **Import errors** — Run `pip install -r requirements.txt` inside the venv

## Project Scripts Reference

| Script | Purpose |
|---|---|
| `scripts/ingest_pipeline.py` | Read Google Sheets, embed, write to Qdrant & Neo4j |
| `scripts/retrieve_and_answer.py` | Answer questions: retrieve + generate |
| `scripts/verify.py` | Verify connections (checks 1, 2, 3) |
| `scripts/evaluate_graphrag.py` | Compare GraphRAG vs vector-only retrieval |
| `scripts/test_neo4j.py` | Quick Neo4j connectivity test (replaces final_check.py) |
| `scripts/test_questions.txt` | 8 questions for batch testing |
| `start_local.ps1` | Start/Quit local services and health-check |
| `app.py` | Streamlit frontend |
| `evaluate_graphrag.py` | Evaluation CLI (GraphRAG vs vector-only) |