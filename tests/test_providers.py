import json

import httpx
import pytest

from regulatory_rag.providers import (
    EmbeddingError,
    OpenAICompatibleEmbeddings,
    validate_embeddings,
)


def test_adapter_sends_request_and_restores_input_order() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://embeddings.example/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-only-key"
        assert json.loads(request.content) == {
            "model": "test-model",
            "input": ["first", "second"],
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ]
            },
        )

    provider = OpenAICompatibleEmbeddings(
        model="test-model",
        base_url="https://embeddings.example/v1/",
        api_key="test-only-key",
        transport=httpx.MockTransport(handle),
    )
    assert provider.embed(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert "test-only-key" not in provider.embedding_id


def test_local_endpoint_can_omit_authentication() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 2]}]})

    provider = OpenAICompatibleEmbeddings(
        base_url="http://localhost:8001/v1", transport=httpx.MockTransport(handle)
    )
    assert provider.embed(["rule"]) == [[1, 2]]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": []},
        {"data": [{"index": 1, "embedding": [1, 2]}]},
        {"data": [{"index": 0, "embedding": []}]},
        {"data": [{"index": 0, "embedding": [0, 0]}]},
        {"data": [{"index": 0, "embedding": ["not-a-number"]}]},
    ],
)
def test_adapter_rejects_malformed_response(payload: dict) -> None:
    provider = OpenAICompatibleEmbeddings(
        base_url="http://localhost/v1",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)),
    )
    with pytest.raises(EmbeddingError):
        provider.embed(["rule"])


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_do_not_expose_response_body(status: int) -> None:
    provider = OpenAICompatibleEmbeddings(
        base_url="http://localhost/v1",
        transport=httpx.MockTransport(
            lambda _: httpx.Response(status, text="sensitive provider output")
        ),
    )
    with pytest.raises(EmbeddingError, match=str(status)) as error:
        provider.embed(["rule"])
    assert "sensitive" not in str(error.value)


def test_timeout_is_reported() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    provider = OpenAICompatibleEmbeddings(
        base_url="http://localhost/v1", transport=httpx.MockTransport(handle)
    )
    with pytest.raises(EmbeddingError, match="Could not reach"):
        provider.embed(["rule"])


def test_empty_input_and_blank_text_do_not_call_provider() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unexpected HTTP request")

    provider = OpenAICompatibleEmbeddings(
        base_url="http://localhost/v1", transport=httpx.MockTransport(handle)
    )
    assert provider.embed([]) == []
    with pytest.raises(EmbeddingError, match="blank"):
        provider.embed([" \n"])


@pytest.mark.parametrize(
    "vectors,count",
    [
        ([], 1),
        ([[1, 0]], 2),
        ([[]], 1),
        ([[0, 0]], 1),
        ([[float("nan"), 1]], 1),
        ([[float("inf"), 1]], 1),
        ([[1, 0], [1, 0, 0]], 2),
    ],
)
def test_invalid_vectors(vectors: list[list[float]], count: int) -> None:
    with pytest.raises(EmbeddingError):
        validate_embeddings(vectors, count)


def test_model_and_endpoint_affect_identity_but_credentials_do_not() -> None:
    first = OpenAICompatibleEmbeddings(base_url="http://localhost/v1", model="a")
    same = OpenAICompatibleEmbeddings(base_url="http://localhost/v1/", model="a", api_key="rotated")
    other_model = OpenAICompatibleEmbeddings(base_url="http://localhost/v1", model="b")
    other_endpoint = OpenAICompatibleEmbeddings(base_url="http://localhost:8000/v1", model="a")
    assert first.embedding_id == same.embedding_id
    assert first.embedding_id != other_model.embedding_id
    assert first.embedding_id != other_endpoint.embedding_id
