import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from regulatory_rag.evaluation import (
    EvaluationCase,
    EvaluationDataset,
    answer_is_grounded,
    console_summary,
    evaluate,
    load_dataset,
    write_report,
)
from regulatory_rag.models import RAGResponse, SourceCitation
from regulatory_rag.service import AnswerExecution


def rag_response(*, status="answered", grounded=True) -> RAGResponse:
    source = {
        "citation_id": "C1",
        "document": "rules.pdf",
        "page": 4,
        "chunk_id": "expected",
        "quotes": ["Annual reports are required."],
    }
    return RAGResponse.model_validate(
        {
            "status": status,
            "answer": "Annual reports are required. [C1]"
            if status == "answered"
            else "Insufficient evidence.",
            "sources": [source] if status == "answered" else [],
            "retrieval_scores": [
                {"chunk_id": "other", "score": 0.8},
                {"chunk_id": "expected", "score": 0.7},
            ],
            "retrieved_chunks": [
                {
                    "chunk_id": "other",
                    "source_document": "other.pdf",
                    "page_number": 1,
                    "text": "Other text.",
                    "score": 0.8,
                },
                {
                    "chunk_id": "expected",
                    "source_document": "rules.pdf",
                    "page_number": 4,
                    "text": "Annual reports are required.",
                    "score": 0.7,
                },
            ],
            "context_chunk_ids": ["other", "expected"],
            "refusal_reason": "model_insufficient" if status != "answered" else None,
        }
    ).model_copy(
        update={
            "sources": []
            if status != "answered" or not grounded
            else [SourceCitation.model_validate(source)]
        }
    )


class FakeService:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def answer_question_timed(self, question, *, top_k):
        self.calls.append((question, top_k))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return AnswerExecution(response, retrieval_seconds=0.012, total_seconds=0.034)


def case(id, *, refusal=False):
    return EvaluationCase(
        id=id,
        question=f"Question {id}",
        expected_refusal=refusal,
        **(
            {}
            if refusal
            else {
                "expected_source": "rules.pdf",
                "expected_page": 4,
                "reference_answer": "Annual reports are required.",
            }
        ),
    )


def test_curated_dataset_has_twenty_unique_balanced_cases():
    dataset = load_dataset(Path(__file__).resolve().parents[1] / "evaluation/dataset.json")
    assert len(dataset.cases) == 20
    assert [item.id for item in dataset.cases] == [f"q{number:03d}" for number in range(1, 21)]
    assert sum(item.expected_refusal for item in dataset.cases) == 4
    assert all(
        item.expected_source and item.expected_page
        for item in dataset.cases
        if not item.expected_refusal
    )


def test_answerable_metrics_groundedness_and_failures_are_all_recorded(tmp_path):
    service = FakeService(
        [rag_response(), rag_response(grounded=False), RuntimeError("provider exploded")]
    )
    report = evaluate(
        service,
        EvaluationDataset(cases=[case("q001"), case("q002"), case("q003")]),
        top_k=2,
        dataset_path="evaluation/dataset.json",
    )
    assert service.calls == [("Question q001", 2), ("Question q002", 2), ("Question q003", 2)]
    assert report.metrics.model_dump() == {
        "total_cases": 3,
        "completed_cases": 2,
        "failed_cases": 1,
        "answerable_cases": 2,
        "refusal_cases": 0,
        "retrieval_hit_at_k": 1.0,
        "document_hit_at_k": 1.0,
        "mean_reciprocal_rank": 0.5,
        "answer_groundedness": 0.5,
        "refusal_accuracy": None,
        "mean_retrieval_latency_ms": 12.0,
        "mean_total_latency_ms": pytest.approx(report.metrics.mean_total_latency_ms),
    }
    assert report.metrics.mean_total_latency_ms is not None
    assert report.metrics.mean_total_latency_ms > 0
    assert report.results[2].outcome == "failed"
    assert report.results[2].error == "RuntimeError: provider exploded"
    json_path, csv_path = write_report(report, tmp_path)
    assert json.loads(json_path.read_text())["results"][2]["outcome"] == "failed"
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 3
    assert rows[2]["outcome"] == "failed"
    assert "Failed cases: q003" in console_summary(report)


def test_refusal_accuracy_and_no_answerable_retrieval_metric():
    service = FakeService([rag_response(status="insufficient_evidence"), rag_response()])
    report = evaluate(
        service,
        EvaluationDataset(cases=[case("q001", refusal=True), case("q002", refusal=True)]),
        top_k=5,
        dataset_path="dataset.json",
    )
    assert report.results[0].refusal_correct is True
    assert report.results[1].refusal_correct is False
    assert report.results[0].retrieval_hit_at_k is None
    assert report.metrics.refusal_accuracy == 0.5
    assert report.metrics.retrieval_hit_at_k is None
    assert report.metrics.mean_reciprocal_rank is None


def test_groundedness_requires_citation_metadata_quotes_and_answer_marker():
    response = rag_response()
    assert answer_is_grounded(response) is True
    assert (
        answer_is_grounded(response.model_copy(update={"answer": "Annual reports are required."}))
        is False
    )
    source = response.sources[0].model_copy(update={"quotes": ["invented"]})
    assert answer_is_grounded(response.model_copy(update={"sources": [source]})) is False
    assert answer_is_grounded(rag_response(status="insufficient_evidence")) is None


@pytest.mark.parametrize(
    "payload",
    [
        [{"id": "q001", "question": "answerable"}],
        [{"id": "q001", "question": "refuse", "expected_refusal": True, "expected_page": 1}],
        [
            {"id": "q001", "question": "one", "expected_refusal": True},
            {"id": "q001", "question": "two", "expected_refusal": True},
        ],
    ],
)
def test_invalid_dataset_cases_are_rejected(tmp_path, payload):
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid evaluation dataset"):
        load_dataset(path)


def test_evaluation_case_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        EvaluationCase(id="q001", question="x", expected_refusal=True, unexpected=True)
