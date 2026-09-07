from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from regulatory_rag import api_client
from regulatory_rag.api_client import APIClientError, query_api
from regulatory_rag.models import RAGResponse

APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def answer() -> RAGResponse:
    return RAGResponse.model_validate(
        {
            "status": "answered",
            "answer": "Submit annual reports. [C1]",
            "sources": [
                {
                    "citation_id": "C1",
                    "document": "rules.pdf",
                    "page": 12,
                    "chunk_id": "reporting",
                    "quotes": ["Submit annual reports."],
                }
            ],
            "retrieval_scores": [{"chunk_id": "reporting", "score": 0.85}],
            "retrieved_chunks": [
                {
                    "chunk_id": "reporting",
                    "source_document": "rules.pdf",
                    "page_number": 12,
                    "text": "Submit annual reports.",
                    "score": 0.85,
                }
            ],
            "context_chunk_ids": ["reporting"],
        }
    )


@pytest.fixture(autouse=True)
def api_environment(monkeypatch):
    monkeypatch.setenv("API_BASE_URL", "http://backend.example:8000/")
    monkeypatch.setenv("API_TIMEOUT_SECONDS", "120")


def test_client_posts_question_and_parses_api_response(answer) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "http://backend.example:8000/query"
        assert request.read() == b'{"question":"Reporting?"}'
        assert "authorization" not in request.headers
        assert request.extensions["timeout"]["read"] == 120
        return httpx.Response(200, json=answer.model_dump())

    assert query_api("Reporting?", transport=httpx.MockTransport(handle)) == answer


@pytest.mark.parametrize(
    "exception,message",
    [
        (httpx.ConnectError, "Cannot connect"),
        (httpx.ReadTimeout, "timed out"),
    ],
)
def test_client_connection_failures(exception, message) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise exception("private transport details", request=request)

    with pytest.raises(APIClientError, match=message) as error:
        query_api("Question", transport=httpx.MockTransport(handle))
    assert "private" not in str(error.value)


def test_client_preserves_useful_api_error() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(
            503,
            json={
                "code": "llm_unavailable",
                "detail": "LLM service is unavailable; try again later",
            },
        )
    )
    with pytest.raises(APIClientError, match="HTTP 503.*try again later"):
        query_api("Question", transport=transport)


@pytest.mark.parametrize("status", [200, 500])
def test_client_handles_non_json_responses_without_exposing_body(status) -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(status, text="private error page"))
    with pytest.raises(APIClientError) as error:
        query_api("Question", transport=transport)
    assert "private error page" not in str(error.value)


@pytest.mark.parametrize(
    "name,value",
    [
        ("API_BASE_URL", "not-a-url"),
        ("API_BASE_URL", "https://user:secret@example.com"),
        ("API_TIMEOUT_SECONDS", "0"),
        ("API_TIMEOUT_SECONDS", "nan"),
    ],
)
def test_client_invalid_configuration_does_not_send_request(monkeypatch, name, value) -> None:
    monkeypatch.setenv(name, value)

    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unexpected API request")

    with pytest.raises(APIClientError, match="Check API_BASE_URL"):
        query_api("Question", transport=httpx.MockTransport(handle))


def test_ui_submits_once_and_displays_answer_citations_and_debug(answer, monkeypatch) -> None:
    request = Mock(return_value=answer)
    monkeypatch.setattr(api_client, "query_api", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    assert not app.exception
    request.assert_not_called()
    app.text_area[0].set_value("  Reporting?  ")
    app.button[0].click().run()
    assert not app.exception
    request.assert_called_once_with("Reporting?")
    texts = [item.value for item in app.text]
    assert "Submit annual reports. [C1]" in texts
    assert "[C1] rules.pdf · Page 12" in texts
    assert any("Similarity: 0.8500" in text and "reporting" in text for text in texts)
    assert app.expander[0].label == "Retrieved chunks and scores"
    assert app.expander[0].proto.expanded is False
    assert not app.warning
    app.run()
    assert request.call_count == 1
    assert "Submit annual reports. [C1]" in [item.value for item in app.text]


def test_ui_insufficient_evidence_is_distinct_and_keeps_debugging(answer, monkeypatch) -> None:
    insufficient = answer.model_copy(
        update={
            "status": "insufficient_evidence",
            "answer": "Insufficient evidence to answer.",
            "sources": [],
            "refusal_reason": "model_insufficient",
        }
    )
    monkeypatch.setattr(api_client, "query_api", Mock(return_value=insufficient))
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("What are the penalties?")
    app.button[0].click().run()
    assert not app.exception
    assert app.warning[0].value == "Insufficient evidence"
    assert "Insufficient evidence to answer." in [item.value for item in app.text]
    assert "No sources cited." in [item.value for item in app.caption]
    assert len(app.expander) == 1
    assert any("Similarity: 0.8500" in item.value for item in app.text)


def test_ui_blank_question_makes_no_api_call(monkeypatch) -> None:
    request = Mock()
    monkeypatch.setattr(api_client, "query_api", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value(" \n")
    app.button[0].click().run()
    assert not app.exception
    assert app.warning[0].value == "Enter a question before submitting."
    request.assert_not_called()


def test_ui_failed_submission_clears_previous_answer(answer, monkeypatch) -> None:
    request = Mock(side_effect=[answer, APIClientError("Cannot connect to the API.")])
    monkeypatch.setattr(api_client, "query_api", request)
    app = AppTest.from_file(str(APP_PATH)).run()
    app.text_area[0].set_value("Reporting?")
    app.button[0].click().run()
    app.text_area[0].set_value("Another question?")
    app.button[0].click().run()
    assert not app.exception
    assert app.error[0].value == "Cannot connect to the API."
    assert not app.expander
    assert "Submit annual reports. [C1]" not in [item.value for item in app.text]
