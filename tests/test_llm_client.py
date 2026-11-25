"""Mocked HTTP unit tests for the handwritten LLM client."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging

import httpx
import pytest
import respx

from app.clients.llm_client import (
    LLMConfigurationError,
    LLMPermanentError,
    LLMResponseParseError,
    LLMRetryExhaustedError,
    build_httpx_timeout,
    complete,
    compute_retry_delay,
    parse_retry_after,
)
from app.config.settings import LLMSettings


class DelayRecorder:
    """Async sleep substitute that records delays without slowing tests."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


@respx.mock
async def test_normal_success_returns_text_usage_and_cost(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Scenario 1: A 200 provider response normalizes text, token usage, and estimated cost."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "choices": [{"message": {"role": "assistant", "content": "Hello from JobFit!"}}],
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            },
        )
    )

    recorder = DelayRecorder()
    result = await complete(
        sample_messages,
        model="gpt-4o-mini",
        max_tokens=128,
        settings=openai_settings,
        sleep_fn=recorder,
    )

    assert route.call_count == 1
    assert recorder.delays == []
    assert result.text == "Hello from JobFit!"
    assert result.model == "gpt-4o-mini"
    assert result.provider == "openai"
    assert result.input_tokens == 1000
    assert result.output_tokens == 500
    assert result.total_tokens == 1500
    assert result.attempts == 1
    assert result.cost_unavailable_reason is None
    # gpt-4o-mini: (1000/1e6 * 0.15) + (500/1e6 * 0.60) = 0.00015 + 0.00030 = 0.00045
    assert result.estimated_cost == pytest.approx(0.00045)
    assert result.pricing_source is not None
    assert result.pricing_last_updated is not None


@respx.mock
async def test_429_then_success_retries_and_respects_retry_after(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Scenario 2: HTTP 429 triggers a retry honoring Retry-After and succeeds on attempt 2."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"error": {"message": "Rate limit exceeded"}},
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Recovered!"}}],
                    "usage": {"prompt_tokens": 200, "completion_tokens": 100},
                },
            ),
        ]
    )

    recorder = DelayRecorder()
    result = await complete(
        sample_messages,
        model="gpt-4o-mini",
        max_tokens=64,
        settings=openai_settings,
        sleep_fn=recorder,
    )

    assert route.call_count == 2
    assert recorder.delays == [3.0]
    assert result.text == "Recovered!"
    assert result.attempts == 2
    assert result.total_tokens == 300


@respx.mock
async def test_repeated_500_errors_raise_after_configured_attempts(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Scenario 3: Three consecutive 500 errors exhaust max_retries=2 (3 total attempts)."""
    settings = openai_settings.model_copy(update={"llm_max_retries": 2})
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(500, json={"error": {"message": "Internal failure 1"}}),
            httpx.Response(500, json={"error": {"message": "Internal failure 2"}}),
            httpx.Response(500, json={"error": {"message": "Internal failure 3"}}),
        ]
    )

    recorder = DelayRecorder()
    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        await complete(
            sample_messages,
            model="gpt-4o-mini",
            max_tokens=64,
            settings=settings,
            sleep_fn=recorder,
        )

    assert route.call_count == 3
    assert len(recorder.delays) == 2
    assert exc_info.value.status_code == 500
    assert exc_info.value.attempts == 3


@respx.mock
async def test_timeout_then_success_retries_and_recovers(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Scenario 4: A ReadTimeout triggers a retry, then a subsequent 200 response succeeds."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.ReadTimeout("Timed out reading provider response"),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "After timeout"}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
                },
            ),
        ]
    )

    recorder = DelayRecorder()
    result = await complete(
        sample_messages,
        model="gpt-4o-mini",
        max_tokens=64,
        settings=openai_settings,
        sleep_fn=recorder,
    )

    assert route.call_count == 2
    assert len(recorder.delays) == 1
    assert 1.0 <= recorder.delays[0] <= 1.5
    assert result.text == "After timeout"
    assert result.attempts == 2


@pytest.mark.parametrize("status_code", [400, 401])
@respx.mock
async def test_permanent_client_errors_are_not_retried(
    status_code: int,
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Scenario 5: HTTP 400 and 401 fail immediately after a single request."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            status_code,
            json={"error": {"message": f"Permanent error {status_code}"}},
        )
    )

    recorder = DelayRecorder()
    with pytest.raises(LLMPermanentError) as exc_info:
        await complete(
            sample_messages,
            model="gpt-4o-mini",
            max_tokens=64,
            settings=openai_settings,
            sleep_fn=recorder,
        )

    assert route.call_count == 1
    assert recorder.delays == []
    assert exc_info.value.status_code == status_code
    assert exc_info.value.attempts == 1


@pytest.mark.parametrize("status_code", [502, 503, 529])
@respx.mock
async def test_other_retryable_statuses_and_anthropic_provider(
    status_code: int,
    anthropic_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Anthropic adapter isolates headers/system message and retries 502/503/529."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=[
            httpx.Response(
                status_code,
                json={"error": {"type": "overloaded_error", "message": "Overloaded"}},
            ),
            httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Anthropic response"}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            ),
        ]
    )

    recorder = DelayRecorder()
    result = await complete(
        sample_messages,
        model="claude-3-5-haiku-latest",
        max_tokens=128,
        settings=anthropic_settings,
        sleep_fn=recorder,
    )

    assert route.call_count == 2
    sent_request = route.calls[0].request
    assert sent_request.headers["x-api-key"] == "sk-ant-test-secret-98765"
    assert sent_request.headers["anthropic-version"] == "2023-06-01"
    assert result.text == "Anthropic response"
    assert result.provider == "anthropic"
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.total_tokens == 150
    assert result.estimated_cost is not None


def test_timeout_configuration_values(openai_settings: LLMSettings) -> None:
    """Explicit connect, read, write, and pool timeouts map from settings."""
    timeout = build_httpx_timeout(openai_settings)
    assert timeout.connect == 5.0
    assert timeout.read == 30.0
    assert timeout.write == 10.0
    assert timeout.pool == 5.0


def test_exponential_backoff_and_jitter_bounds(
    openai_settings: LLMSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backoff follows 1s, 2s, 4s, 8s base progression with bounded jitter."""
    monkeypatch.setattr("app.clients.llm_client.random.uniform", lambda a, b: b)
    delays = [compute_retry_delay(attempt, openai_settings) for attempt in range(1, 5)]
    assert delays == [
        (1.5, "exponential-backoff"),
        (2.5, "exponential-backoff"),
        (4.5, "exponential-backoff"),
        (8.5, "exponential-backoff"),
    ]


def test_retry_after_parsing_and_clamping() -> None:
    """Retry-After supports numeric seconds, HTTP dates, clamping, and rejects invalid values."""
    assert parse_retry_after("4", max_wait=16.0) == 4.0
    assert parse_retry_after("999", max_wait=16.0) == 16.0
    assert parse_retry_after("-2", max_wait=16.0) is None
    assert parse_retry_after("not-a-date", max_wait=16.0) is None

    now = datetime(2026, 11, 25, 12, 0, 0, tzinfo=timezone.utc)
    future_http_date = (now + timedelta(seconds=7)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert parse_retry_after(future_http_date, max_wait=16.0, now=now) == 7.0


@respx.mock
async def test_missing_usage_or_pricing_reports_explicit_unavailable_cost(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """When usage or model pricing is missing, estimated_cost is None with a clear reason."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "No usage"}}]},
            ),
            httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "Unknown model"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                },
            ),
        ]
    )

    no_usage = await complete(sample_messages, model="gpt-4o-mini", max_tokens=32, settings=openai_settings)
    assert no_usage.estimated_cost is None
    assert no_usage.cost_unavailable_reason is not None

    unknown_model = await complete(
        sample_messages,
        model="custom-unpriced-model",
        max_tokens=32,
        settings=openai_settings,
    )
    assert unknown_model.estimated_cost is None
    assert unknown_model.cost_unavailable_reason is not None


@respx.mock
async def test_malformed_provider_response_raises_parse_error(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
) -> None:
    """Malformed JSON or unexpected schema raises LLMResponseParseError without retrying."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"unexpected": "shape"})
    )

    with pytest.raises(LLMResponseParseError):
        await complete(sample_messages, model="gpt-4o-mini", max_tokens=32, settings=openai_settings)
    assert route.call_count == 1


async def test_missing_api_key_fails_before_http_call(
    sample_messages: list[dict[str, str]],
) -> None:
    """Missing API key raises LLMConfigurationError immediately."""
    settings = LLMSettings(llm_api_key=None)
    with pytest.raises(LLMConfigurationError):
        await complete(sample_messages, model="gpt-4o-mini", max_tokens=32, settings=settings)


@respx.mock
async def test_logs_and_errors_never_expose_api_key_or_prompt_content(
    openai_settings: LLMSettings,
    sample_messages: list[dict[str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Operational logs include attempt/status/latency while keeping secrets and prompts out."""
    secret = openai_settings.llm_api_key.get_secret_value()  # type: ignore[union-attr]
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"Retry-After": "1"},
                json={"error": {"message": f"Bad rate limit echoing {secret}"}},
            ),
            httpx.Response(
                401,
                json={"error": {"message": f"Invalid key {secret}"}},
            ),
        ]
    )

    recorder = DelayRecorder()
    with caplog.at_level(logging.INFO, logger="app.clients.llm_client"):
        with pytest.raises(LLMPermanentError) as exc_info:
            await complete(
                sample_messages,
                model="gpt-4o-mini",
                max_tokens=32,
                settings=openai_settings,
                sleep_fn=recorder,
            )

    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)

    combined_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert "model=gpt-4o-mini" in combined_logs
    assert "attempt=1/5" in combined_logs
    assert "status=429" in combined_logs
    assert "latency_ms=" in combined_logs
    assert "delay_s=" in combined_logs
    assert secret not in combined_logs
    assert "SENSITIVE_PROMPT_SECRET_DO_NOT_LOG" not in combined_logs
