"""Minimal presentation layer over POST /query. Run with streamlit run."""

import streamlit as st

from regulatory_rag.api_client import APIClientError, query_api
from regulatory_rag.models import RAGResponse


def display_result(result: RAGResponse) -> None:
    st.subheader("Answer")
    if result.status == "insufficient_evidence":
        st.warning("Insufficient evidence")
    # Render API text literally so source text cannot introduce HTML or Markdown images.
    st.text(result.answer)

    st.subheader("Sources")
    if not result.sources:
        st.caption("No sources cited.")
    for source in result.sources:
        page = f"Page {source.page}" if source.page is not None else "Page not available"
        st.text(f"[{source.citation_id}] {source.document} · {page}")
        for quote in source.quotes:
            st.text(f"“{quote}”")

    with st.expander("Retrieved chunks and scores", expanded=False):
        st.caption("Similarity scores indicate retrieval relevance, not answer confidence.")
        if result.refusal_reason:
            st.text(f"Response reason: {result.refusal_reason}")
        if not result.retrieved_chunks:
            st.text("No chunks were retrieved.")
        selected = set(result.context_chunk_ids)
        for index, chunk in enumerate(result.retrieved_chunks, start=1):
            if index > 1:
                st.divider()
            page = (
                f"Page {chunk.page_number}"
                if chunk.page_number is not None
                else "Page not available"
            )
            st.text(f"{index}. {chunk.source_document} · {page}")
            st.text(f"Similarity: {chunk.score:.4f} · Chunk ID: {chunk.chunk_id}")
            st.caption(
                "Included in context" if chunk.chunk_id in selected else "Not included in context"
            )
            st.text(chunk.text)


def main() -> None:
    st.set_page_config(page_title="Regulatory RAG Assistant", layout="centered")
    st.title("Regulatory RAG Assistant")
    st.caption("Ask a question about your indexed regulatory documents.")

    with st.form("question_form"):
        question = st.text_area(
            "Question", placeholder="What reporting obligations apply?", max_chars=10000, height=110
        )
        submitted = st.form_submit_button("Submit", type="primary")

    if submitted:
        st.session_state.pop("rag_result", None)
        if not question.strip():
            st.warning("Enter a question before submitting.")
        else:
            with st.spinner("Looking for an answer…"):
                try:
                    st.session_state["rag_result"] = query_api(question.strip())
                except APIClientError as exc:
                    st.error(str(exc))

    result = st.session_state.get("rag_result")
    if isinstance(result, RAGResponse):
        display_result(result)


if __name__ == "__main__":
    main()
