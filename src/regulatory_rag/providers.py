"""Replaceable embedding/LLM interfaces and OpenAI-compatible HTTP adapters."""

import hashlib
import math
from typing import Literal, Protocol

import httpx
from pydantic import BaseModel, Field, ValidationError

from regulatory_rag.config import LLMConfig


class LLMProvider(Protocol):
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return completed text or raise LLMError; no retrieval or prompt policy."""
        ...


class LLMError(RuntimeError):
    """Provider-independent failure without request bodies or credentials."""

    def __init__(self, message: str, *, code: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class _ChatMessage(BaseModel):
    role: Literal["assistant"]
    content: str | None = None
    refusal: str | None = None


class _ChatChoice(BaseModel):
    index: int = Field(ge=0, strict=True)
    message: _ChatMessage
    finish_reason: str


class _ChatResponse(BaseModel):
    choices: list[_ChatChoice] = Field(min_length=1, max_length=1)


class OpenAICompatibleLLM:
    """Synchronous, non-streaming chat completions with no automatic retries."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config if config is not None else LLMConfig.from_env()
        self._transport = transport

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not user_prompt.strip():
            raise ValueError("user_prompt must not be blank")
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        headers = {}
        if self.config.api_key is not None:
            headers["Authorization"] = f"Bearer {self.config.api_key.get_secret_value()}"
        timeout = httpx.Timeout(
            self.config.timeout_seconds, connect=self.config.connect_timeout_seconds
        )
        try:
            with httpx.Client(
                timeout=timeout, transport=self._transport, follow_redirects=False
            ) as client:
                response = client.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json={"model": self.config.model, "messages": messages, "stream": False},
                )
                response.raise_for_status()
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out", code="timeout") from None
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in {401, 403}:
                code, message = "authentication", "LLM authentication or permission denied"
            elif status == 429:
                code, message = "rate_limit", "LLM rate limit or quota exceeded"
            elif status >= 500:
                code, message = "unavailable", "LLM service is unavailable"
            else:
                code, message = "http_error", "LLM request was rejected"
            raise LLMError(f"{message} (HTTP {status})", code=code, status_code=status) from None
        except httpx.RequestError:
            raise LLMError("Could not reach the LLM service", code="connection") from None
        try:
            completion = _ChatResponse.model_validate_json(response.content)
        except ValidationError:
            raise LLMError(
                "LLM returned an invalid chat response", code="invalid_response"
            ) from None
        choice = completion.choices[0]
        if choice.index != 0:
            raise LLMError("LLM returned an unexpected choice index", code="invalid_response")
        if choice.message.refusal or choice.finish_reason == "content_filter":
            raise LLMError("LLM declined to produce a response", code="refusal")
        if choice.finish_reason == "length":
            raise LLMError("LLM response was truncated", code="truncated")
        if choice.finish_reason != "stop":
            raise LLMError("LLM did not return completed text", code="invalid_response")
        content = choice.message.content
        if content is None or not content.strip():
            raise LLMError("LLM returned empty text", code="invalid_response")
        return content


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
