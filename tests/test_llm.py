import json
import traceback

import httpx
import pytest
from pydantic import ValidationError

from regulatory_rag.config import LLMConfig
from regulatory_rag.providers import LLMError, LLMProvider, OpenAICompatibleLLM


@pytest.fixture(autouse=True)
def clean_llm_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "LLM_MODEL",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "OPENAI_API_KEY",
        "LLM_TIMEOUT_SECONDS",
        "LLM_CONNECT_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)


def completion(content: str = "Generated text", finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ]
    }


def make_provider(handler) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        LLMConfig(model="test-chat", base_url="http://localhost:8000/v1"),
        transport=httpx.MockTransport(handler),
    )


def test_chat_request_authentication_timeouts_and_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "test-chat")
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.example/api/v1/")
    monkeypatch.setenv("LLM_API_KEY", "test-only-secret")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_CONNECT_TIMEOUT_SECONDS", "2")

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://llm.example/api/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-only-secret"
        assert request.extensions["timeout"] == {
            "connect": 2.0,
            "read": 12.5,
            "write": 12.5,
            "pool": 12.5,
        }
        assert json.loads(request.content) == {
            "model": "test-chat",
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain a rule."},
            ],
            "stream": False,
        }
        return httpx.Response(200, json=completion("  A rule.\n"))

    provider: LLMProvider = OpenAICompatibleLLM(transport=httpx.MockTransport(handle))
    assert provider.generate("Be concise.", "Explain a rule.") == "  A rule.\n"


def test_local_endpoint_omits_auth_and_empty_system_prompt() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        assert json.loads(request.content)["messages"] == [{"role": "user", "content": "Hello"}]
        assert request.extensions["timeout"] == {
            "connect": 5.0,
            "read": 60.0,
            "write": 60.0,
            "pool": 60.0,
        }
        return httpx.Response(200, json=completion())

    assert make_provider(handle).generate(" \n", "Hello") == "Generated text"


def test_protocol_can_be_implemented_without_inheritance() -> None:
    class FakeLLM:
        def generate(self, system_prompt: str, user_prompt: str) -> str:
            return "Offline response"

    provider: LLMProvider = FakeLLM()
    assert provider.generate("system", "question") == "Offline response"


def test_environment_fallback_precedence_and_call_time_loading(monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "first")
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-test-key")
    first = LLMConfig.from_env()
    assert first.api_key.get_secret_value() == "fallback-test-key"
    monkeypatch.setenv("LLM_API_KEY", "specific-test-key")
    monkeypatch.setenv("LLM_MODEL", "second")
    second = LLMConfig.from_env()
    assert second.api_key.get_secret_value() == "specific-test-key"
    assert first.model == "first"
    assert second.model == "second"


def test_explicit_configuration_does_not_require_environment() -> None:
    config = LLMConfig(model="local", base_url="http://localhost/v1")
    assert OpenAICompatibleLLM(config).config is config


def test_missing_model_and_hosted_credentials_rejected(monkeypatch) -> None:
    with pytest.raises(ValidationError):
        LLMConfig.from_env()
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with pytest.raises(ValidationError, match="LLM_API_KEY"):
        OpenAICompatibleLLM()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "bad", ""])
@pytest.mark.parametrize("name", ["LLM_TIMEOUT_SECONDS", "LLM_CONNECT_TIMEOUT_SECONDS"])
def test_invalid_environment_timeouts(name, value, monkeypatch) -> None:
    monkeypatch.setenv("LLM_MODEL", "local")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost/v1")
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        LLMConfig.from_env()


@pytest.mark.parametrize(
    "url",
    [
        "ftp://localhost/v1",
        "not-a-url",
        "https://",
        "http://[invalid",
        "https://user:password@llm.example/v1",
        "https://llm.example/v1?key=secret",
        "https://llm.example/v1#fragment",
    ],
)
def test_invalid_or_credential_bearing_urls_rejected(url) -> None:
    with pytest.raises(ValidationError):
        LLMConfig(model="test-model", base_url=url)


@pytest.mark.parametrize("key", ["has spaces", "line\nbreak", "non-ascii-é", "null\x00byte"])
def test_invalid_auth_tokens_rejected_without_echoing(key) -> None:
    with pytest.raises(ValidationError) as error:
        LLMConfig(model="test-model", base_url="http://localhost/v1", api_key=key)
    assert key not in str(error.value)


def test_secret_excluded_from_representations_and_config_errors() -> None:
    key = "test-only-do-not-expose"
    config = LLMConfig(model="test-model", api_key=key)
    assert key not in repr(config)
    assert "api_key" not in config.model_dump()
    assert key not in config.model_dump_json()
    with pytest.raises(ValidationError) as error:
        LLMConfig(model="test-model", api_key=key, timeout_seconds=-1)
    assert key not in str(error.value)


@pytest.mark.parametrize("prompt", ["", " \n\t"])
def test_blank_user_prompt_rejected_without_request(prompt) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        pytest.fail("Unexpected HTTP call")

    with pytest.raises(ValueError, match="user_prompt"):
        make_provider(handle).generate("system", prompt)


@pytest.mark.parametrize(
    "status,code",
    [
        (400, "http_error"),
        (401, "authentication"),
        (403, "authentication"),
        (404, "http_error"),
        (429, "rate_limit"),
        (500, "unavailable"),
        (503, "unavailable"),
        (302, "http_error"),
    ],
)
def test_http_errors_are_safe_and_not_retried_or_redirected(status, code) -> None:
    calls = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            text="sensitive-response-body",
            headers={
                "location": "https://unexpected.example/",
            },
        )

    with pytest.raises(LLMError) as error:
        make_provider(handle).generate("system", "private-prompt")
    assert error.value.code == code
    assert error.value.status_code == status
    rendered = "".join(traceback.format_exception(error.value))
    assert "sensitive-response-body" not in rendered
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure,code",
    [
        (httpx.ConnectTimeout, "timeout"),
        (httpx.ReadTimeout, "timeout"),
        (httpx.WriteTimeout, "timeout"),
        (httpx.PoolTimeout, "timeout"),
        (httpx.ConnectError, "connection"),
        (httpx.RemoteProtocolError, "connection"),
    ],
)
def test_transport_errors_are_wrapped(failure, code) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise failure("sensitive-transport-detail", request=request)

    with pytest.raises(LLMError) as error:
        make_provider(handle).generate("system", "user")
    assert error.value.code == code
    assert error.value.status_code is None
    assert "sensitive-transport-detail" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": "text"}}]},
        completion(""),
        completion(" \n"),
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "user", "content": "text"},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 1,
                    "message": {"role": "assistant", "content": "text"},
                    "finish_reason": "stop",
                }
            ]
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ["text"]},
                    "finish_reason": "stop",
                }
            ]
        },
    ],
)
def test_malformed_or_empty_completions_rejected(payload) -> None:
    provider = make_provider(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(LLMError) as error:
        provider.generate("system", "user")
    assert error.value.code == "invalid_response"


def test_non_json_response_rejected() -> None:
    provider = make_provider(lambda _: httpx.Response(200, text="not-json-private-content"))
    with pytest.raises(LLMError) as error:
        provider.generate("system", "user")
    assert error.value.code == "invalid_response"
    assert "not-json-private-content" not in str(error.value)


@pytest.mark.parametrize(
    "reason,code",
    [
        ("length", "truncated"),
        ("content_filter", "refusal"),
        ("tool_calls", "invalid_response"),
        ("function_call", "invalid_response"),
        ("unknown", "invalid_response"),
    ],
)
def test_incomplete_completions_are_not_returned_as_success(reason, code) -> None:
    provider = make_provider(lambda _: httpx.Response(200, json=completion("partial", reason)))
    with pytest.raises(LLMError) as error:
        provider.generate("system", "user")
    assert error.value.code == code


def test_explicit_refusal_is_reported() -> None:
    payload = completion()
    payload["choices"][0]["message"]["refusal"] = "refusal text"
    provider = make_provider(lambda _: httpx.Response(200, json=payload))
    with pytest.raises(LLMError) as error:
        provider.generate("system", "user")
    assert error.value.code == "refusal"
