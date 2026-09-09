# Run with: streamlit run app.py
"""
Business Legal Assistant — Streamlit Demo App
Chat interface for the GraphRAG retrieval + generation pipeline.
"""

import streamlit as st
import sys
import os
from typing import List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))
from retrieve_and_answer import answer_question


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Business Legal Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Minimal CSS — only for citation pills (under 20 lines)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.cite-pill {
    display: inline-block;
    background: #2F2F2F;
    border: 1px solid #3A3A3A;
    border-radius: 12px;
    padding: 4px 10px;
    font-size: 0.8rem;
    color: #B0B0B0;
    margin: 2px 4px 2px 0;
    white-space: nowrap;
}
.cite-pill:hover {
    border-color: #10A37F;
    color: #ECECEC;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_example_questions() -> List[str]:
    qfile = os.path.join(os.path.dirname(__file__), "scripts", "test_questions.txt")
    if os.path.exists(qfile):
        with open(qfile, "r", encoding="utf-8") as f:
            questions = [q.strip() for q in f if q.strip()]
        if questions:
            return questions
    return [
        "Vendor breached a contract — can damages be claimed?",
        "An MSME supplier wasn't paid within 45 days — what can they do?",
        "Can an LLP partner be held personally liable for the firm's debts?",
        "Is a builder liable to a homebuyer for delayed possession under consumer law?",
        "Can an employer terminate a female employee during her maternity leave?",
        "What legal recourse exists if an employee leaks company data before resigning?",
    ]


EXAMPLE_QUESTIONS = load_example_questions()


def render_citations(sources_used: List[Tuple[str, str, str]]) -> str:
    if not sources_used:
        return ""
    chips = []
    for title, sec_num, source_url in sources_used:
        if source_url:
            chips.append(
                f'<span class="cite-pill">'
                f'{title} &middot; Sec {sec_num}'
                f'</span>'
            )
        else:
            chips.append(
                f'<span class="cite-pill">'
                f'{title} &middot; Sec {sec_num}'
                f'</span>'
            )
    return '<div style="margin-top:8px;">' + "".join(chips) + '</div>'


def render_trace(raw_context: dict):
    vector_hits = raw_context.get("vector_hits", [])
    graph_context = raw_context.get("graph_context", [])
    if not vector_hits and not graph_context:
        return
    with st.expander("Show retrieval details"):
        if vector_hits:
            st.markdown("**Vector search results**")
            for i, hit in enumerate(vector_hits[:5], 1):
                p = hit.get("payload", {})
                t = p.get("title", "?")
                s = p.get("section_number", "")
                sc = hit.get("score")
                sc_str = f"{sc:.4f}" if sc is not None else "N/A"
                st.text(f"#{i}  score={sc_str}  {t}, Section {s}")
        if graph_context:
            st.markdown("**Graph-expanded context**")
            for i, ctx in enumerate(graph_context[:8], 1):
                t = ctx.get("title", "?")
                s = ctx.get("section_number", "")
                why = ctx.get("why", "")
                st.text(f"#{i}  {why}  {t}, Section {s}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.markdown(
            "A GraphRAG system that combines vector search with knowledge graph traversal "
            "to answer business-legal questions grounded in verified Indian law sources."
        )
        st.markdown("**Examples**")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(q, key=f"sb_{i}", use_container_width=True):
                st.session_state["example_question"] = q
        st.caption("19 documents · 127 provisions\n\nContract · Employment · Consumer")

    # --- Session state ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Title ---
    st.title("Business Legal Assistant")
    st.caption("Answers grounded in Indian Acts & Judgments")

    # --- Handle example question click ---
    example_q = st.session_state.pop("example_question", None)

    # --- Replay chat history ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if sources:
                    st.markdown(render_citations(sources), unsafe_allow_html=True)
                raw_ctx = msg.get("raw_context", {})
                if raw_ctx:
                    render_trace(raw_ctx)

    # --- Empty state ---
    if not st.session_state.messages and not example_q:
        st.info("Ask a question about Indian business law to get started.")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLE_QUESTIONS[:4]):
            col = cols[i % 2]
            with col:
                if st.button(q, key=f"ex_{i}", use_container_width=True):
                    st.session_state["example_question"] = q
                    st.rerun()

    # --- Process example question ---
    if example_q:
        st.session_state.messages.append({"role": "user", "content": example_q})
        with st.chat_message("user"):
            st.markdown(example_q)

        with st.chat_message("assistant"):
            with st.spinner("Searching legal database..."):
                try:
                    result = answer_question(example_q)
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources_used", [])
                    raw_ctx = result.get("raw_context", {})
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []
                    raw_ctx = {}

            st.markdown(answer)
            if sources:
                st.markdown(render_citations(sources), unsafe_allow_html=True)
            if raw_ctx:
                render_trace(raw_ctx)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "raw_context": raw_ctx,
        })

    # --- Chat input ---
    user_question = st.chat_input("Ask a business-legal question...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching legal database..."):
                try:
                    result = answer_question(user_question)
                    answer = result.get("answer", "No answer returned.")
                    sources = result.get("sources_used", [])
                    raw_ctx = result.get("raw_context", {})
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    sources = []
                    raw_ctx = {}

            st.markdown(answer)
            if sources:
                st.markdown(render_citations(sources), unsafe_allow_html=True)
            if raw_ctx:
                render_trace(raw_ctx)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "raw_context": raw_ctx,
        })


if __name__ == "__main__":
    main()
