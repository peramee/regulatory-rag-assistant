"""Replaceable embedding interface and an OpenAI-compatible HTTP adapter."""

import hashlib
import math
from typing import Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError


class EmbeddingError(ValueError):
    """The embedding service failed or returned unusable vectors."""


class EmbeddingProvider(Protocol):
    @property
    def embedding_id(self) -> str:
        """Stable identity of the model, endpoint, and embedding configuration."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one finite, nonzero, fixed-size vector per text, in input order."""
        ...


def validate_embeddings(vectors: list[list[float]], expected_count: int) -> int:
    """Validate all vectors before storage and return their shared dimension."""
    if len(vectors) != expected_count or not vectors:
        raise EmbeddingError("Embedding count does not match the requested texts")
    dimension = len(vectors[0])
    if dimension == 0:
        raise EmbeddingError("Embeddings must not be empty")
    for vector in vectors:
        if len(vector) != dimension:
            raise EmbeddingError("Embedding dimensions are inconsistent")
        if not all(math.isfinite(value) for value in vector) or not any(vector):
            raise EmbeddingError("Embeddings must contain finite values and be nonzero")
    return dimension


class _EmbeddingItem(BaseModel):
    index: int = Field(ge=0, strict=True)
    embedding: list[float]


class _EmbeddingResponse(BaseModel):
    data: list[_EmbeddingItem]


class OpenAICompatibleEmbeddings:
    """Call a /embeddings endpoint; API keys are supplied by the caller.

    An optional HTTP transport makes the real adapter testable without network
    access. Local endpoints may omit authentication. No credentials are persisted.
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        url = httpx.URL(base_url)
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("Embedding base URL must be an HTTP(S) URL")
        if url.userinfo or url.query or url.fragment:
            raise ValueError("Embedding base URL must not contain credentials, query, or fragment")
        if not model.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("A model name and a positive finite timeout are required")
        if url.host == "api.openai.com" and not api_key:
            raise ValueError("Set EMBEDDING_API_KEY for the OpenAI embedding endpoint")
        self.model = model
        self.base_url = str(url).rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport

    @property
    def embedding_id(self) -> str:
        endpoint_hash = hashlib.sha256(self.base_url.encode()).hexdigest()[:16]
        return f"openai-compatible:{endpoint_hash}:{self.model}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed blank text")
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            with httpx.Client(timeout=self._timeout, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json={"model": self.model, "input": texts, "encoding_format": "float"},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Avoid exposing provider response bodies, documents, or credentials.
            raise EmbeddingError(
                f"Embedding service returned HTTP {exc.response.status_code}"
            ) from None
        except httpx.RequestError:
            raise EmbeddingError("Could not reach the embedding service") from None
        try:
            data = _EmbeddingResponse.model_validate_json(response.content).data
        except ValidationError:
            raise EmbeddingError("Embedding service returned an invalid response") from None
        ordered = sorted(data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise EmbeddingError("Embedding response indices do not match the requested texts")
        vectors = [item.embedding for item in ordered]
        validate_embeddings(vectors, len(texts))
        return vectors
