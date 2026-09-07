"""Validated LLM configuration, loaded explicitly from the environment."""

import os
from typing import Self
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator


class LLMConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    model: str = Field(min_length=1)
    base_url: str = "https://api.openai.com/v1"
    api_key: SecretStr | None = Field(default=None, repr=False, exclude=True)
    timeout_seconds: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    connect_timeout_seconds: float = Field(default=5.0, gt=0, allow_inf_nan=False)

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("LLM_MODEL must not be blank")
        return value.strip()

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        try:
            # Reject malformed bracketed hosts that HTTPX's URL parser permits.
            urlsplit(value)
            url = httpx.URL(value)
        except (httpx.InvalidURL, ValueError):
            raise ValueError("LLM_BASE_URL must be a valid HTTP(S) base URL") from None
        if url.scheme not in {"http", "https"} or not url.host:
            raise ValueError("LLM_BASE_URL must be a valid HTTP(S) base URL")
        if url.userinfo or url.query or url.fragment:
            raise ValueError("LLM_BASE_URL must not contain credentials, query, or fragment")
        return str(url).rstrip("/")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        key = value.get_secret_value()
        if not key.strip():
            return None
        if not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
            raise ValueError("LLM_API_KEY must be an ASCII token without whitespace")
        return value

    @model_validator(mode="after")
    def require_hosted_credentials(self) -> Self:
        if httpx.URL(self.base_url).host == "api.openai.com" and self.api_key is None:
            raise ValueError("Set LLM_API_KEY or OPENAI_API_KEY for the OpenAI endpoint")
        return self

    @classmethod
    def from_env(cls) -> Self:
        """Read at call time; .env files are not automatically loaded."""
        return cls.model_validate(
            {
                "model": os.environ.get("LLM_MODEL", ""),
                "base_url": os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
                "api_key": os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                "timeout_seconds": os.environ.get("LLM_TIMEOUT_SECONDS", "60"),
                "connect_timeout_seconds": os.environ.get("LLM_CONNECT_TIMEOUT_SECONDS", "5"),
            }
        )
