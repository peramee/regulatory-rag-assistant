"""End-to-end question answering with grounded output and retrieval diagnostics."""

import logging
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from threading import RLock
from time import perf_counter
from typing import BinaryIO
from uuid import uuid4

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
from regulatory_rag.observability import safe_error_details
from regulatory_rag.providers import LLMProvider
from regulatory_rag.retrieval import Retriever

INSUFFICIENT_EVIDENCE = "Insufficient evidence in the retrieved documents to answer this question."
logger = logging.getLogger("regulatory_rag.rag")


class DocumentUploadError(ValueError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class AnswerExecution:
    """An answer and elapsed timings for evaluation or operational instrumentation."""

    def __init__(
        self,
        response: RAGResponse,
        retrieval_seconds: float,
        total_seconds: float,
        *,
        request_id: str | None = None,
        llm_seconds: float | None = None,
        approximate_input_tokens: int | None = None,
        approximate_output_tokens: int | None = None,
    ) -> None:
        self.response = response
        self.retrieval_seconds = retrieval_seconds
        self.total_seconds = total_seconds
        self.request_id = request_id
        self.llm_seconds = llm_seconds
        self.approximate_input_tokens = approximate_input_tokens
        self.approximate_output_tokens = approximate_output_tokens


class _GenerationTelemetry:
    def __init__(self) -> None:
        self.llm_seconds: float | None = None
        self.input_characters: int | None = None
        self.output_characters: int | None = None


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

    def index_directory(self, directory: str | Path) -> int:
        """Index supported files below a directory, skipping downloaded originals."""
        root = Path(directory)
        if not root.is_dir():
            return 0
        total = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".pdf"}:
                continue
            if "originals" in path.parts:
                continue
            with path.open("rb") as content:
                result = self.index_document(path.name, content)
            total += result.new_chunks
        return total

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

    def answer_question(
        self, question: str, *, top_k: int | None = None, request_id: str | None = None
    ) -> RAGResponse:
        return self.answer_question_timed(question, top_k=top_k, request_id=request_id).response

    def answer_question_timed(
        self, question: str, *, top_k: int | None = None, request_id: str | None = None
    ) -> AnswerExecution:
        """Answer once while separating retrieval time from total service time."""
        if not question.strip():
            raise ValueError("Question must not be blank")
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k must be a positive integer")
        request_id = request_id or uuid4().hex
        started = perf_counter()
        retrieval_seconds: float | None = None
        retrieved: list[SearchResult] = []
        telemetry = _GenerationTelemetry()
        try:
            with self._lock:
                retrieval_started = perf_counter()
                retrieved = self.retriever.search(question, top_k=top_k or self.config.top_k)
                retrieval_seconds = perf_counter() - retrieval_started
                response = self._answer_from_retrieved(question, retrieved, telemetry)
            total_seconds = perf_counter() - started
            execution = AnswerExecution(
                response=response,
                retrieval_seconds=retrieval_seconds,
                total_seconds=total_seconds,
                request_id=request_id,
                llm_seconds=telemetry.llm_seconds,
                approximate_input_tokens=_estimate_tokens(telemetry.input_characters),
                approximate_output_tokens=_estimate_tokens(telemetry.output_characters),
            )
            self._log_lifecycle("rag_request_completed", execution, retrieved)
            return execution
        except Exception as error:
            self._log_failure(
                request_id=request_id,
                retrieval_seconds=retrieval_seconds,
                llm_seconds=telemetry.llm_seconds,
                total_seconds=perf_counter() - started,
                retrieved=retrieved,
                error=error,
            )
            raise

    def _answer_from_retrieved(
        self,
        question: str,
        retrieved: list[SearchResult],
        telemetry: _GenerationTelemetry,
    ) -> RAGResponse:
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
        telemetry.input_characters = len(system_prompt) + len(user_prompt)
        llm_started = perf_counter()
        try:
            raw = self.llm.generate(system_prompt, user_prompt)
        finally:
            telemetry.llm_seconds = perf_counter() - llm_started
        telemetry.output_characters = len(raw)
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

    def _log_lifecycle(
        self, event: str, execution: AnswerExecution, retrieved: list[SearchResult]
    ) -> None:
        logger.info(
            event,
            extra={
                "event": event,
                "request_id": execution.request_id,
                "retrieval_latency_ms": _milliseconds(execution.retrieval_seconds),
                "llm_latency_ms": _milliseconds(execution.llm_seconds),
                "total_latency_ms": _milliseconds(execution.total_seconds),
                "chunks_retrieved": len(retrieved),
                "retrieval_scores": _scores(retrieved),
                "model": _model_name(self.llm),
                "approximate_input_tokens": execution.approximate_input_tokens,
                "approximate_output_tokens": execution.approximate_output_tokens,
                "refused": execution.response.status == "insufficient_evidence",
                "error_type": None,
                "error_code": None,
            },
        )

    def _log_failure(
        self,
        *,
        request_id: str,
        retrieval_seconds: float | None,
        llm_seconds: float | None,
        total_seconds: float,
        retrieved: list[SearchResult],
        error: Exception,
    ) -> None:
        logger.error(
            "rag_request_failed",
            extra={
                "event": "rag_request_failed",
                "request_id": request_id,
                "retrieval_latency_ms": _milliseconds(retrieval_seconds),
                "llm_latency_ms": _milliseconds(llm_seconds),
                "total_latency_ms": _milliseconds(total_seconds),
                "chunks_retrieved": len(retrieved),
                "retrieval_scores": _scores(retrieved),
                "model": _model_name(self.llm),
                "approximate_input_tokens": None,
                "approximate_output_tokens": None,
                "refused": None,
                **safe_error_details(error),
            },
        )


def _milliseconds(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 3)


def _estimate_tokens(characters: int | None) -> int | None:
    return None if characters is None else max(1, round(characters / 4))


def _scores(retrieved: list[SearchResult]) -> list[dict[str, float | str]]:
    return [{"chunk_id": item.chunk_id, "score": item.score} for item in retrieved]


def _model_name(llm: LLMProvider) -> str:
    model = getattr(getattr(llm, "config", None), "model", None)
    return model if isinstance(model, str) and model else type(llm).__name__
