import io
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from regulatory_rag.models import SearchResult
from regulatory_rag.observability import JsonFormatter
from regulatory_rag.providers import LLMError, LLMProvider
from regulatory_rag.retrieval import Retriever
from regulatory_rag.service import RAGService, logger


def passage() -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        source_document="rules.pdf",
        page_number=7,
        text="Operators must submit transaction reports.",
        score=0.91,
    )


def completed_answer() -> str:
    return json.dumps(
        {
            "status": "answered",
            "claims": [
                {
                    "text": "Operators must submit transaction reports.",
                    "evidence": [
                        {
                            "source_id": "C1",
                            "quote": "Operators must submit transaction reports.",
                        }
                    ],
                }
            ],
        }
    )


def logged_payloads():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def make_service(results: list[SearchResult], output: str) -> RAGService:
    retriever = Mock(spec=Retriever)
    retriever.search.return_value = results
    llm = Mock(spec=LLMProvider)
    llm.config = SimpleNamespace(model="telemetry-test-model")
    llm.generate.return_value = output
    return RAGService(retriever, llm)


def test_completed_request_logs_safe_json_lifecycle_metadata() -> None:
    service = make_service([passage()], completed_answer())
    generator = logged_payloads()
    stream = next(generator)
    try:
        execution = service.answer_question_timed(
            "Which operators report? confidential-question", request_id="request-123"
        )
    finally:
        next(generator, None)

    payload = json.loads(stream.getvalue())
    assert payload["event"] == "rag_request_completed"
    assert payload["request_id"] == "request-123"
    assert payload["chunks_retrieved"] == 1
    assert payload["retrieval_scores"] == [{"chunk_id": "chunk-1", "score": 0.91}]
    assert payload["model"] == "telemetry-test-model"
    assert payload["refused"] is False
    assert payload["approximate_input_tokens"] > 0
    assert payload["approximate_output_tokens"] > 0
    assert payload["retrieval_latency_ms"] >= 0
    assert payload["llm_latency_ms"] >= 0
    assert payload["total_latency_ms"] >= 0
    assert execution.request_id == "request-123"
    assert "confidential-question" not in stream.getvalue()
    assert "Operators must submit" not in stream.getvalue()


def test_refusal_and_provider_failure_are_logged_without_error_messages() -> None:
    service = make_service([], "unused")
    generator = logged_payloads()
    stream = next(generator)
    try:
        service.answer_question_timed("unanswerable", request_id="refusal-id")
    finally:
        next(generator, None)
    refusal = json.loads(stream.getvalue())
    assert refusal["refused"] is True
    assert refusal["llm_latency_ms"] is None
    assert refusal["approximate_input_tokens"] is None

    service = make_service([passage()], "unused")
    service.llm.generate.side_effect = LLMError("secret provider payload", code="timeout")
    generator = logged_payloads()
    stream = next(generator)
    try:
        with pytest.raises(LLMError):
            service.answer_question_timed("private question", request_id="failure-id")
    finally:
        next(generator, None)
    failure = json.loads(stream.getvalue())
    assert failure["event"] == "rag_request_failed"
    assert failure["error_type"] == "LLMError"
    assert failure["error_code"] == "timeout"
    assert failure["llm_latency_ms"] >= 0
    assert failure["refused"] is None
    assert "secret provider payload" not in stream.getvalue()
    assert "private question" not in stream.getvalue()


def test_json_formatter_ignores_unapproved_log_record_fields() -> None:
    record = logging.LogRecord("test", logging.INFO, __file__, 0, "event", (), None)
    record.event = "safe_event"
    record.api_key = "not-for-logs"
    assert "not-for-logs" not in JsonFormatter().format(record)
