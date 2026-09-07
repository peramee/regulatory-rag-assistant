"""End-to-end question answering with grounded output and retrieval diagnostics."""

from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from threading import RLock
from typing import BinaryIO

from regulatory_rag.config import RAGConfig
from regulatory_rag.generation import (
    GroundingError,
    build_grounded_prompt,
    parse_grounded_output,
    render_grounded_answer,
)
from regulatory_rag.ingestion import ingest_document
from regulatory_rag.models import (
    ChunkingConfig,
    DocumentListResponse,
    IndexingResponse,
    RAGResponse,
    RetrievalScore,
    SearchResult,
)
from regulatory_rag.providers import LLMProvider
from regulatory_rag.retrieval import Retriever

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the retrieved documents to answer this question."


class DocumentUploadError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class RAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        config: RAGConfig | None = None,
        *,
        chunking: ChunkingConfig | None = None,
        max_upload_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or RAGConfig()
        self.chunking = chunking or ChunkingConfig()
        if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be a positive integer")
        self.max_upload_bytes = max_upload_bytes
        self._lock = RLock()

    def index_document(self, filename: str, content: BinaryIO) -> IndexingResponse:
        """Stage an upload under a generated path; retain only its source basename."""
        # PureWindowsPath understands both slash styles, including browser fake paths.
        source = PureWindowsPath(filename).name
        if not source.strip() or any(ord(character) < 32 for character in source):
            raise DocumentUploadError("Provide a valid source filename", code="invalid_filename")
        suffix = Path(source).suffix.lower()
        if suffix not in {".txt", ".pdf"}:
            raise DocumentUploadError(
                "Supported document types are .txt and .pdf", code="unsupported_type"
            )
        with TemporaryDirectory(prefix="regulatory-upload-") as directory:
            staged = Path(directory) / f"upload{suffix}"
            size = 0
            with staged.open("wb") as output:
                while block := content.read(min(65536, self.max_upload_bytes - size + 1)):
                    size += len(block)
                    if size > self.max_upload_bytes:
                        raise DocumentUploadError(
                            f"Document exceeds the {self.max_upload_bytes}-byte upload limit",
                            code="too_large",
                        )
                    output.write(block)
            chunks = ingest_document(staged, self.chunking, source_document=source)
        with self._lock:
            before = self.retriever.store.count()
            indexed = self.retriever.index(chunks)
            added = self.retriever.store.count() - before
        return IndexingResponse(document=source, chunks_indexed=indexed, new_chunks=added)

    def list_documents(self) -> DocumentListResponse:
        with self._lock:
            return DocumentListResponse(documents=self.retriever.store.list_documents())

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

    def answer_question(self, question: str, *, top_k: int | None = None) -> RAGResponse:
        if not question.strip():
            raise ValueError("Question must not be blank")
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        with self._lock:
            return self._answer_question(question, top_k=top_k or self.config.top_k)

    def _answer_question(self, question: str, *, top_k: int) -> RAGResponse:
        retrieved = self.retriever.search(question, top_k=top_k)
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
