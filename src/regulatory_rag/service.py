"""End-to-end question answering with grounded output and retrieval diagnostics."""

from regulatory_rag.config import RAGConfig
from regulatory_rag.generation import (
    GroundingError,
    build_grounded_prompt,
    parse_grounded_output,
    render_grounded_answer,
)
from regulatory_rag.models import RAGResponse, RetrievalScore, SearchResult
from regulatory_rag.providers import LLMProvider
from regulatory_rag.retrieval import Retriever

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the retrieved documents to answer this question."


class RAGService:
    def __init__(
        self, retriever: Retriever, llm: LLMProvider, config: RAGConfig | None = None
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or RAGConfig()

    def _select_context(self, retrieved: list[SearchResult]) -> list[SearchResult]:
        remaining = self.config.max_context_chars
        selected: list[SearchResult] = []
        seen: set[str] = set()
        for chunk in retrieved:
            if chunk.chunk_id in seen:
                continue
            if self.config.min_score is not None and chunk.score < self.config.min_score:
                continue
            if len(chunk.text) > remaining:
                continue
            selected.append(chunk)
            seen.add(chunk.chunk_id)
            remaining -= len(chunk.text)
        return selected

    def answer_question(self, question: str) -> RAGResponse:
        if not question.strip():
            raise ValueError("Question must not be blank")
        retrieved = self.retriever.search(question, top_k=self.config.top_k)
        context = self._select_context(retrieved)
        response = RAGResponse(
            status="insufficient_evidence",
            answer=INSUFFICIENT_EVIDENCE,
            sources=[],
            retrieval_scores=[
                RetrievalScore(chunk_id=chunk.chunk_id, score=chunk.score) for chunk in retrieved
            ],
            retrieved_chunks=retrieved,
            context_chunk_ids=[chunk.chunk_id for chunk in context],
            refusal_reason="no_context",
        )
        if not context:
            return response

        system_prompt, user_prompt = build_grounded_prompt(question, context)
        # Provider/transport errors propagate, rather than masquerading as a lack of evidence.
        raw = self.llm.generate(system_prompt, user_prompt)
        try:
            output = parse_grounded_output(raw, context)
        except GroundingError:
            return response.model_copy(update={"refusal_reason": "invalid_model_output"})
        if output.status == "insufficient_evidence":
            return response.model_copy(update={"refusal_reason": "model_insufficient"})
        answer, sources = render_grounded_answer(output, context)
        return response.model_copy(
            update={
                "status": "answered",
                "answer": answer,
                "sources": sources,
                "refusal_reason": None,
            }
        )
