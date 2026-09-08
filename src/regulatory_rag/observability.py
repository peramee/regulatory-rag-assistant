"""Small, privacy-conscious JSON logging helpers for RAG request telemetry."""

import json
import logging
from datetime import UTC, datetime
from typing import Any

LOG_FIELD_NAMES = (
    "event",
    "request_id",
    "retrieval_latency_ms",
    "llm_latency_ms",
    "total_latency_ms",
    "chunks_retrieved",
    "retrieval_scores",
    "model",
    "approximate_input_tokens",
    "approximate_output_tokens",
    "refused",
    "refusal_reason",
    "error_type",
    "error_code",
)


class JsonFormatter(logging.Formatter):
    """Render an allow-listed telemetry record as one JSON object.

    Keeping an explicit field list prevents accidental logging of prompts,
    document text, API keys, or arbitrary third-party exception details.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for name in LOG_FIELD_NAMES:
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_structured_logging(level: str | int = "INFO") -> None:
    """Configure the application logger once without changing root logging."""
    logger = logging.getLogger("regulatory_rag")
    logger.setLevel(level)
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


def safe_error_details(error: Exception) -> dict[str, str]:
    """Return stable error metadata without including an exception message."""
    code = getattr(error, "code", None)
    return {
        "error_type": type(error).__name__,
        "error_code": code if isinstance(code, str) else "unexpected_error",
    }
