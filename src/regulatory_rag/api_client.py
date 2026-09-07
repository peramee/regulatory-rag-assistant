"""HTTP-only frontend client; shared response schemas, no RAG business logic."""

import math
import os
from urllib.parse import urlsplit

import httpx
from pydantic import ValidationError

from regulatory_rag.models import ErrorResponse, RAGResponse


class APIClientError(RuntimeError):
    """A user-facing API connection, configuration, or response error."""


def query_api(question: str, *, transport: httpx.BaseTransport | None = None) -> RAGResponse:
    """Submit a question to FastAPI; configuration is read at request time."""
    try:
        base_url = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
        urlsplit(base_url)
        url = httpx.URL(base_url)
        timeout = float(os.environ.get("API_TIMEOUT_SECONDS", "120"))
        if (
            url.scheme not in {"http", "https"}
            or not url.host
            or url.userinfo
            or url.query
            or url.fragment
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("Invalid API configuration")
    except (httpx.InvalidURL, ValueError):
        raise APIClientError(
            "Check API_BASE_URL and API_TIMEOUT_SECONDS in the frontend environment."
        ) from None

    try:
        with httpx.Client(
            timeout=httpx.Timeout(timeout, connect=5.0), transport=transport, follow_redirects=False
        ) as client:
            response = client.post(f"{str(url).rstrip('/')}/query", json={"question": question})
            response.raise_for_status()
    except httpx.TimeoutException:
        raise APIClientError("The request timed out. Please try again shortly.") from None
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        try:
            detail = ErrorResponse.model_validate_json(exc.response.content).detail
        except ValidationError:
            detail = "The API could not complete the request. Please try again."
        raise APIClientError(f"Request failed (HTTP {status}): {detail}") from None
    except httpx.RequestError:
        raise APIClientError(
            "Cannot connect to the API. Check that the backend is running."
        ) from None
    try:
        return RAGResponse.model_validate_json(response.content)
    except ValidationError:
        raise APIClientError("The API returned an unexpected response. Please try again.") from None
