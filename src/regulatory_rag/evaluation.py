"""Repeatable, failure-preserving evaluation over the existing RAG service."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from regulatory_rag.generation import source_quote_match
from regulatory_rag.models import RAGResponse, SearchResult
from regulatory_rag.service import RAGService


class EvaluationCase(BaseModel):
    """One manually reviewed answerable or deliberately unanswerable question."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^q[0-9]{3}$")
    question: str = Field(min_length=1, pattern=r"\S")
    expected_source: str | None = None
    expected_page: int | None = Field(default=None, ge=1)
    expected_pages: list[int] = Field(default_factory=list)
    reference_answer: str | None = None
    expected_refusal: bool = False

    @model_validator(mode="after")
    def validate_expectations(self) -> "EvaluationCase":
        answerable = not self.expected_refusal
        has_expected_evidence = self.expected_source is not None and self.expected_page is not None
        if answerable and (not has_expected_evidence or not self.reference_answer):
            raise ValueError(
                "Answerable cases require expected_source, expected_page, and reference_answer"
            )
        if self.expected_refusal and (
            self.expected_source is not None
            or self.expected_page is not None
            or self.expected_pages
            or self.reference_answer is not None
        ):
            raise ValueError("Refusal cases cannot define expected evidence or a reference_answer")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "EvaluationDataset":
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Evaluation case IDs must be unique")
        return self


class CaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    expected_refusal: bool
    expected_source: str | None
    expected_page: int | None
    expected_pages: list[int] = Field(default_factory=list)
    reference_answer: str | None
    outcome: Literal["completed", "failed"]
    error: str | None = None
    retrieval_hit_at_k: bool | None = None
    document_hit_at_k: bool | None = None
    reciprocal_rank: float | None = Field(default=None, ge=0, le=1)
    answer_status: Literal["answered", "insufficient_evidence"] | None = None
    answer_grounded: bool | None = None
    refusal_correct: bool | None = None
    retrieval_latency_ms: float | None = Field(default=None, ge=0)
    total_latency_ms: float = Field(ge=0)
    answer: str | None = None
    sources: list[dict[str, object]] = []
    retrieved_chunks: list[dict[str, object]] = []
    context_chunk_ids: list[str] = []
    refusal_reason: str | None = None


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total_cases: int
    completed_cases: int
    failed_cases: int
    answerable_cases: int
    refusal_cases: int
    retrieval_hit_at_k: float | None
    document_hit_at_k: float | None
    mean_reciprocal_rank: float | None
    answer_groundedness: float | None
    refusal_accuracy: float | None
    answerable_coverage: float | None
    false_refusal_rate: float | None
    unsafe_answer_rate: float | None
    mean_retrieval_latency_ms: float | None
    mean_total_latency_ms: float | None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: str
    top_k: int
    dataset_path: str
    metrics: AggregateMetrics
    results: list[CaseResult]


def load_dataset(path: str | Path) -> EvaluationDataset:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        # Accept the requested list format while keeping room for future metadata.
        return (
            EvaluationDataset(cases=payload)
            if isinstance(payload, list)
            else EvaluationDataset.model_validate(payload)
        )
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid evaluation dataset: {exc}") from None


def _rank(case: EvaluationCase, chunks: list[SearchResult]) -> tuple[int | None, int | None]:
    document_rank = next(
        (
            index
            for index, chunk in enumerate(chunks, start=1)
            if chunk.source_document == case.expected_source
        ),
        None,
    )
    exact_rank = next(
        (
            index
            for index, chunk in enumerate(chunks, start=1)
            if chunk.source_document == case.expected_source
            and chunk.page_number
            in (set(case.expected_pages) or ({case.expected_page} if case.expected_page else set()))
        ),
        None,
    )
    return exact_rank, document_rank


def answer_is_grounded(response: RAGResponse) -> bool | None:
    """Verify returned citations/quotes against retrieved chunks, not entailment."""
    if response.status != "answered":
        return None
    chunks = {chunk.chunk_id: chunk for chunk in response.retrieved_chunks}
    if not response.answer.strip() or not response.sources:
        return False
    for source in response.sources:
        chunk = chunks.get(source.chunk_id)
        if (
            chunk is None
            or source.document != chunk.source_document
            or source.page != chunk.page_number
            or not source.quotes
            or any(source_quote_match(quote, chunk.text) is None for quote in source.quotes)
            or f"[{source.citation_id}]" not in response.answer
        ):
            return False
    return True


def evaluate_case(service: RAGService, case: EvaluationCase, *, top_k: int) -> CaseResult:
    started = perf_counter()
    try:
        execution = service.answer_question_timed(case.question, top_k=top_k)
        response = execution.response
    except Exception as exc:
        return CaseResult(
            id=case.id,
            question=case.question,
            expected_refusal=case.expected_refusal,
            expected_source=case.expected_source,
            expected_page=case.expected_page,
            expected_pages=case.expected_pages,
            reference_answer=case.reference_answer,
            outcome="failed",
            error=f"{type(exc).__name__}: {exc}",
            total_latency_ms=(perf_counter() - started) * 1000,
        )
    exact_rank, document_rank = _rank(case, response.retrieved_chunks)
    expected_status = "insufficient_evidence" if case.expected_refusal else "answered"
    refusal_correct = response.status == expected_status
    return CaseResult(
        id=case.id,
        question=case.question,
        expected_refusal=case.expected_refusal,
        expected_source=case.expected_source,
        expected_page=case.expected_page,
        expected_pages=case.expected_pages,
        reference_answer=case.reference_answer,
        outcome="completed",
        retrieval_hit_at_k=exact_rank is not None if not case.expected_refusal else None,
        document_hit_at_k=document_rank is not None if not case.expected_refusal else None,
        reciprocal_rank=(1 / exact_rank)
        if exact_rank is not None
        else (0 if not case.expected_refusal else None),
        answer_status=response.status,
        answer_grounded=answer_is_grounded(response),
        refusal_correct=refusal_correct,
        retrieval_latency_ms=execution.retrieval_seconds * 1000,
        total_latency_ms=execution.total_seconds * 1000,
        answer=response.answer,
        sources=[source.model_dump() for source in response.sources],
        retrieved_chunks=[chunk.model_dump() for chunk in response.retrieved_chunks],
        context_chunk_ids=response.context_chunk_ids,
        refusal_reason=response.refusal_reason,
    )


def aggregate(results: list[CaseResult]) -> AggregateMetrics:
    def mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    completed = [result for result in results if result.outcome == "completed"]
    answerable = [result for result in completed if not result.expected_refusal]
    refusals = [result for result in completed if result.expected_refusal]
    return AggregateMetrics(
        total_cases=len(results),
        completed_cases=len(completed),
        failed_cases=len(results) - len(completed),
        answerable_cases=len(answerable),
        refusal_cases=len(refusals),
        retrieval_hit_at_k=mean(
            [
                float(result.retrieval_hit_at_k)
                for result in answerable
                if result.retrieval_hit_at_k is not None
            ]
        ),
        document_hit_at_k=mean(
            [
                float(result.document_hit_at_k)
                for result in answerable
                if result.document_hit_at_k is not None
            ]
        ),
        mean_reciprocal_rank=mean(
            [result.reciprocal_rank for result in answerable if result.reciprocal_rank is not None]
        ),
        answer_groundedness=mean(
            [
                float(result.answer_grounded)
                for result in completed
                if result.answer_grounded is not None
            ]
        ),
        refusal_accuracy=mean(
            [
                float(result.refusal_correct)
                for result in completed
                if result.refusal_correct is not None
            ]
        ),
        answerable_coverage=mean([float(r.answer_status == "answered") for r in answerable]),
        false_refusal_rate=mean(
            [float(r.answer_status == "insufficient_evidence") for r in answerable]
        ),
        unsafe_answer_rate=mean([float(r.answer_status == "answered") for r in refusals]),
        mean_retrieval_latency_ms=mean(
            [
                result.retrieval_latency_ms
                for result in completed
                if result.retrieval_latency_ms is not None
            ]
        ),
        mean_total_latency_ms=mean([result.total_latency_ms for result in results]),
    )


def evaluate(
    service: RAGService, dataset: EvaluationDataset, *, top_k: int, dataset_path: str
) -> EvaluationReport:
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    results = [evaluate_case(service, case, top_k=top_k) for case in dataset.cases]
    return EvaluationReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        top_k=top_k,
        dataset_path=dataset_path,
        metrics=aggregate(results),
        results=results,
    )


def write_report(report: EvaluationReport, output_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "results.json"
    csv_path = directory / "results.csv"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    rows = [
        {
            "id": result.id,
            "question": result.question,
            "expected_refusal": result.expected_refusal,
            "expected_source": result.expected_source,
            "expected_page": result.expected_page,
            "expected_pages": ";".join(str(page) for page in result.expected_pages),
            "outcome": result.outcome,
            "error": result.error,
            "retrieval_hit_at_k": result.retrieval_hit_at_k,
            "document_hit_at_k": result.document_hit_at_k,
            "reciprocal_rank": result.reciprocal_rank,
            "answer_status": result.answer_status,
            "answer_grounded": result.answer_grounded,
            "refusal_correct": result.refusal_correct,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "total_latency_ms": result.total_latency_ms,
            "answer": result.answer,
            "source_chunk_ids": ";".join(str(source["chunk_id"]) for source in result.sources),
            "retrieved_chunk_ids": ";".join(
                str(chunk["chunk_id"]) for chunk in result.retrieved_chunks
            ),
            "refusal_reason": result.refusal_reason,
        }
        for result in report.results
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def console_summary(report: EvaluationReport) -> str:
    metrics = report.metrics

    def format_metric(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.3f}"

    failed_ids = (
        ", ".join(result.id for result in report.results if result.outcome == "failed") or "none"
    )
    return "\n".join(
        [
            (
                f"Evaluation: {metrics.completed_cases}/{metrics.total_cases} completed; "
                f"failed: {metrics.failed_cases}"
            ),
            (
                f"Retrieval hit@{report.top_k}: {format_metric(metrics.retrieval_hit_at_k)}; "
                f"document hit@{report.top_k}: {format_metric(metrics.document_hit_at_k)}; "
                f"MRR: {format_metric(metrics.mean_reciprocal_rank)}"
            ),
            (
                f"Groundedness: {format_metric(metrics.answer_groundedness)}; "
                f"refusal accuracy: {format_metric(metrics.refusal_accuracy)}; "
                f"false refusal rate: {format_metric(metrics.false_refusal_rate)}"
            ),
            (
                f"Mean retrieval latency: {format_metric(metrics.mean_retrieval_latency_ms)} ms; "
                f"mean total latency: {format_metric(metrics.mean_total_latency_ms)} ms"
            ),
            f"Failed cases: {failed_ids}",
        ]
    )
