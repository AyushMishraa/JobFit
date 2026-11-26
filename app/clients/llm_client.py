"""Handwritten asynchronous HTTP client for calling LLM APIs reliably."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import logging
import math
import random
import time
from typing import Any

import httpx

from app.config.settings import LLMSettings, get_settings
from app.schemas.llm import LLMMessage, LLMResponse, ModelPricing, TokenUsage

logger = logging.getLogger(__name__)

# HTTP statuses that represent temporary server/rate-limit conditions worth retrying.
RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 529})

# Transient transport exceptions raised by httpx before a complete HTTP response arrives.
TRANSIENT_EXCEPTIONS: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)

SleepFn = Callable[[float], Awaitable[None]]


class LLMClientError(Exception):
    """Base exception for all LLM client failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
        model: str | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider
        self.model = model
        self.attempts = attempts


class LLMConfigurationError(LLMClientError):
    """Raised when required client settings (such as API key or messages) are invalid."""


class LLMPermanentError(LLMClientError):
    """Raised immediately on non-retryable errors such as HTTP 400 or 401."""


class LLMRetryExhaustedError(LLMClientError):
    """Raised when all retry attempts for transient failures are exhausted."""


class LLMResponseParseError(LLMClientError):
    """Raised when a provider returns malformed JSON or an unexpected payload shape."""


@dataclass(frozen=True)
class _ProviderSpec:
    """Pairs isolated request building and response parsing for one provider."""

    build_request: Callable[
        [str, str, str, list[LLMMessage], int],
        tuple[str, dict[str, str], dict[str, Any]],
    ]
    parse_response: Callable[[dict[str, Any]], tuple[str, TokenUsage]]


def _redact_secret(text: str, secret: str) -> str:
    """Replace any accidental occurrence of an API key before logging or raising."""
    if secret and secret in text:
        return text.replace(secret, "[REDACTED]")
    return text


def _normalize_messages(
    messages: Sequence[LLMMessage | Mapping[str, str]],
) -> list[LLMMessage]:
    """Validate and convert raw message dicts into :class:`LLMMessage` models."""
    if not messages:
        raise LLMConfigurationError("At least one message is required.")
    normalized: list[LLMMessage] = []
    for msg in messages:
        if isinstance(msg, LLMMessage):
            normalized.append(msg)
        else:
            normalized.append(LLMMessage.model_validate(msg))
    return normalized


def _build_chat_completions_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[LLMMessage],
    max_tokens: int,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build URL, headers, and JSON body for Mistral / OpenAI-compatible Chat Completions."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "max_tokens": max_tokens,
    }
    return url, headers, payload


def _extract_chat_message_text(content: Any) -> str:
    """Extract assistant text from either a plain string or Mistral content chunk list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            chunk["text"]
            for chunk in content
            if isinstance(chunk, dict)
            and chunk.get("type") == "text"
            and isinstance(chunk.get("text"), str)
        ]
        if text_parts:
            return "".join(text_parts)
    raise LLMResponseParseError(
        "Provider response is missing valid text in 'choices[0].message.content'."
    )


def _parse_chat_completions_response(payload: dict[str, Any]) -> tuple[str, TokenUsage]:
    """Extract generated text and normalized token usage from a Chat Completions response."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMResponseParseError("Provider response is missing a non-empty 'choices' list.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise LLMResponseParseError("Provider response 'choices[0]' must be an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise LLMResponseParseError("Provider response is missing 'choices[0].message' object.")

    text = _extract_chat_message_text(message.get("content"))

    usage_raw = payload.get("usage")
    if not isinstance(usage_raw, dict):
        return text, TokenUsage()

    input_tokens = _as_optional_int(usage_raw.get("prompt_tokens"))
    output_tokens = _as_optional_int(usage_raw.get("completion_tokens"))
    total_tokens = _as_optional_int(usage_raw.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return text, TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _build_anthropic_request(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[LLMMessage],
    max_tokens: int,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build URL, headers, and JSON body for the Anthropic Messages API."""
    url = f"{base_url}/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    system_parts = [m.content for m in messages if m.role == "system"]
    chat_messages = [
        {"role": m.role, "content": m.content}
        for m in messages
        if m.role != "system"
    ]
    if not chat_messages:
        raise LLMConfigurationError(
            "Anthropic requests require at least one non-system message."
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": chat_messages,
        "max_tokens": max_tokens,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    return url, headers, payload


def _parse_anthropic_response(payload: dict[str, Any]) -> tuple[str, TokenUsage]:
    """Extract generated text and normalized token usage from an Anthropic response."""
    blocks = payload.get("content")
    if not isinstance(blocks, list) or not blocks:
        raise LLMResponseParseError("Provider response is missing a non-empty 'content' list.")

    text_parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text_value = block.get("text")
            if isinstance(text_value, str):
                text_parts.append(text_value)

    if not text_parts:
        raise LLMResponseParseError("Provider response 'content' contained no text blocks.")

    usage_raw = payload.get("usage")
    if not isinstance(usage_raw, dict):
        return "".join(text_parts), TokenUsage()

    input_tokens = _as_optional_int(usage_raw.get("input_tokens"))
    output_tokens = _as_optional_int(usage_raw.get("output_tokens"))
    total_tokens = (
        input_tokens + output_tokens
        if input_tokens is not None and output_tokens is not None
        else None
    )
    return "".join(text_parts), TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


_PROVIDERS: dict[str, _ProviderSpec] = {
    "mistral": _ProviderSpec(
        build_request=_build_chat_completions_request,
        parse_response=_parse_chat_completions_response,
    ),
    "openai": _ProviderSpec(
        build_request=_build_chat_completions_request,
        parse_response=_parse_chat_completions_response,
    ),
    "anthropic": _ProviderSpec(
        build_request=_build_anthropic_request,
        parse_response=_parse_anthropic_response,
    ),
}


def _as_optional_int(value: Any) -> int | None:
    """Return a non-negative integer if valid, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _extract_error_detail(response: httpx.Response, secret: str) -> str:
    """Safely summarize an HTTP error body without exposing request headers or secrets."""
    detail = ""
    try:
        data = response.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or err.get("type") or "")
            elif isinstance(err, str):
                detail = err
            elif "message" in data:
                detail = str(data["message"])
            elif "detail" in data:
                detail = str(data["detail"])
    except ValueError:
        detail = response.text.strip()

    detail = " ".join(detail.split())[:300]
    sanitized = _redact_secret(detail, secret)
    return sanitized or response.reason_phrase or f"HTTP {response.status_code}"


def parse_retry_after(
    value: str | None,
    *,
    max_wait: float,
    now: datetime | None = None,
) -> float | None:
    """Parse an HTTP ``Retry-After`` header (seconds or HTTP-date), bounded by ``max_wait``.

    Returns ``None`` if the header is missing, negative, or unparseable.
    """
    if not value:
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    # Form 1: delta-seconds (e.g. "2" or "1.5")
    try:
        seconds = float(cleaned)
    except ValueError:
        seconds = None

    if seconds is not None:
        if math.isnan(seconds) or math.isinf(seconds) or seconds < 0.0:
            return None
        return min(seconds, max_wait)

    # Form 2: IMF-fixdate / HTTP-date (e.g. "Wed, 21 Oct 2015 07:28:00 GMT")
    try:
        target_dt = parsedate_to_datetime(cleaned)
    except (TypeError, ValueError):
        return None

    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    reference_now = now or datetime.now(timezone.utc)
    delta = (target_dt - reference_now).total_seconds()
    return max(0.0, min(delta, max_wait))


def compute_retry_delay(
    attempt: int,
    settings: LLMSettings,
    retry_after_header: str | None = None,
) -> tuple[float, str]:
    """Return ``(delay_seconds, reason)`` respecting ``Retry-After`` or bounded backoff+jitter."""
    retry_after = parse_retry_after(
        retry_after_header,
        max_wait=settings.llm_backoff_max_seconds,
    )
    if retry_after is not None:
        return retry_after, "retry-after"

    base_delay = min(
        settings.llm_backoff_base_seconds * (2 ** (attempt - 1)),
        settings.llm_backoff_max_seconds,
    )
    jitter = random.uniform(0.0, settings.llm_jitter_max_seconds)
    bounded_delay = min(base_delay + jitter, settings.llm_backoff_max_seconds)
    return bounded_delay, "exponential-backoff"


def calculate_cost(
    usage: TokenUsage,
    pricing: ModelPricing | None,
) -> tuple[float | None, str | None]:
    """Compute estimated USD cost, or return an explicit reason when unavailable."""
    if pricing is None:
        return None, "No pricing configured for this model."
    if usage.input_tokens is None or usage.output_tokens is None:
        return None, "Provider response omitted input or output token usage."

    cost = (
        (usage.input_tokens / 1_000_000) * pricing.input_price_per_million
        + (usage.output_tokens / 1_000_000) * pricing.output_price_per_million
    )
    return round(cost, 8), None


def build_httpx_timeout(settings: LLMSettings) -> httpx.Timeout:
    """Create an explicit :class:`httpx.Timeout` from settings."""
    return httpx.Timeout(
        connect=settings.llm_connect_timeout,
        read=settings.llm_read_timeout,
        write=settings.llm_write_timeout,
        pool=settings.llm_pool_timeout,
    )


class LLMClient:
    """Asynchronous LLM HTTP client with timeouts, bounded retries, logging, and cost tracking."""

    def __init__(
        self,
        settings: LLMSettings | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        sleep_fn: SleepFn = asyncio.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._http_client = http_client
        self._sleep_fn = sleep_fn

    async def complete(
        self,
        messages: Sequence[LLMMessage | Mapping[str, str]],
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a completion request and return a normalized :class:`LLMResponse`."""
        normalized_messages = _normalize_messages(messages)
        resolved_model = model or self.settings.llm_default_model
        resolved_max_tokens = (
            max_tokens if max_tokens is not None else self.settings.llm_default_max_tokens
        )
        if resolved_max_tokens <= 0:
            raise LLMConfigurationError("max_tokens must be greater than zero.")

        api_key = (
            self.settings.llm_api_key.get_secret_value().strip()
            if self.settings.llm_api_key is not None
            else ""
        )
        if not api_key:
            raise LLMConfigurationError(
                "LLM API key is missing. Set MISTRAL_API_KEY or LLM_API_KEY in the environment or settings."
            )

        provider = self.settings.llm_provider
        spec = _PROVIDERS.get(provider)
        if spec is None:
            raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")

        url, headers, payload = spec.build_request(
            self.settings.resolved_base_url,
            api_key,
            resolved_model,
            normalized_messages,
            resolved_max_tokens,
        )
        timeout = build_httpx_timeout(self.settings)

        if self._http_client is not None:
            return await self._execute_with_retries(
                client=self._http_client,
                spec=spec,
                provider=provider,
                model=resolved_model,
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                api_key=api_key,
            )

        # Open a scoped client so connections are closed cleanly even on timeouts.
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await self._execute_with_retries(
                client=client,
                spec=spec,
                provider=provider,
                model=resolved_model,
                url=url,
                headers=headers,
                payload=payload,
                timeout=timeout,
                api_key=api_key,
            )

    async def _execute_with_retries(
        self,
        *,
        client: httpx.AsyncClient,
        spec: _ProviderSpec,
        provider: str,
        model: str,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: httpx.Timeout,
        api_key: str,
    ) -> LLMResponse:
        max_attempts = 1 + self.settings.llm_max_retries
        total_start = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            attempt_start = time.perf_counter()
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            except TRANSIENT_EXCEPTIONS as exc:
                latency_ms = round((time.perf_counter() - attempt_start) * 1000, 2)
                reason = type(exc).__name__
                logger.warning(
                    "LLM request transient error model=%s attempt=%d/%d status=None latency_ms=%.2f error_type=%s",
                    model,
                    attempt,
                    max_attempts,
                    latency_ms,
                    reason,
                )
                if attempt == max_attempts:
                    logger.error(
                        "LLM request failed after exhausting retries model=%s attempts=%d status=None reason=%s",
                        model,
                        attempt,
                        reason,
                    )
                    raise LLMRetryExhaustedError(
                        f"LLM request failed after {attempt} attempt(s) due to {reason}.",
                        provider=provider,
                        model=model,
                        attempts=attempt,
                    ) from exc

                delay, delay_source = compute_retry_delay(attempt, self.settings)
                logger.info(
                    "Retrying LLM request model=%s next_attempt=%d/%d delay_s=%.3f source=%s reason=%s",
                    model,
                    attempt + 1,
                    max_attempts,
                    delay,
                    delay_source,
                    reason,
                )
                await self._sleep_fn(delay)
                continue
            except httpx.HTTPError as exc:
                latency_ms = round((time.perf_counter() - attempt_start) * 1000, 2)
                reason = type(exc).__name__
                logger.error(
                    "LLM request permanent transport error model=%s attempt=%d/%d latency_ms=%.2f error_type=%s",
                    model,
                    attempt,
                    max_attempts,
                    latency_ms,
                    reason,
                )
                raise LLMPermanentError(
                    f"Non-retryable HTTP client error ({reason}).",
                    provider=provider,
                    model=model,
                    attempts=attempt,
                ) from exc

            latency_ms = round((time.perf_counter() - attempt_start) * 1000, 2)
            status_code = response.status_code

            if 200 <= status_code < 300:
                return self._build_success_response(
                    response=response,
                    spec=spec,
                    provider=provider,
                    model=model,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    attempt_latency_ms=latency_ms,
                    total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
                )

            error_detail = _extract_error_detail(response, api_key)

            if status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    "LLM request retryable HTTP status model=%s attempt=%d/%d status=%d latency_ms=%.2f",
                    model,
                    attempt,
                    max_attempts,
                    status_code,
                    latency_ms,
                )
                if attempt == max_attempts:
                    logger.error(
                        "LLM request failed after exhausting retries model=%s attempts=%d status=%d",
                        model,
                        attempt,
                        status_code,
                    )
                    raise LLMRetryExhaustedError(
                        f"LLM request failed with HTTP {status_code} after {attempt} attempt(s): {error_detail}",
                        status_code=status_code,
                        provider=provider,
                        model=model,
                        attempts=attempt,
                    )

                delay, delay_source = compute_retry_delay(
                    attempt,
                    self.settings,
                    response.headers.get("Retry-After"),
                )
                logger.info(
                    "Retrying LLM request model=%s next_attempt=%d/%d delay_s=%.3f source=%s reason=HTTP_%d",
                    model,
                    attempt + 1,
                    max_attempts,
                    delay,
                    delay_source,
                    status_code,
                )
                await self._sleep_fn(delay)
                continue

            # Permanent client/API error (e.g., 400, 401, 403, 404): fail immediately.
            logger.error(
                "LLM request permanent failure model=%s attempt=%d/%d status=%d latency_ms=%.2f",
                model,
                attempt,
                max_attempts,
                status_code,
                latency_ms,
            )
            raise LLMPermanentError(
                f"LLM provider rejected request with HTTP {status_code}: {error_detail}",
                status_code=status_code,
                provider=provider,
                model=model,
                attempts=attempt,
            )

        raise AssertionError("Unreachable: retry loop exited without returning or raising.")

    def _build_success_response(
        self,
        *,
        response: httpx.Response,
        spec: _ProviderSpec,
        provider: str,
        model: str,
        attempt: int,
        max_attempts: int,
        attempt_latency_ms: float,
        total_latency_ms: float,
    ) -> LLMResponse:
        try:
            payload = response.json()
        except ValueError as exc:
            logger.error(
                "LLM response invalid JSON model=%s attempt=%d/%d status=%d latency_ms=%.2f",
                model,
                attempt,
                max_attempts,
                response.status_code,
                attempt_latency_ms,
            )
            raise LLMResponseParseError(
                "Provider returned invalid JSON.",
                status_code=response.status_code,
                provider=provider,
                model=model,
                attempts=attempt,
            ) from exc

        if not isinstance(payload, dict):
            raise LLMResponseParseError(
                "Provider JSON payload must be an object.",
                status_code=response.status_code,
                provider=provider,
                model=model,
                attempts=attempt,
            )

        try:
            text, usage = spec.parse_response(payload)
        except LLMResponseParseError as exc:
            raise LLMResponseParseError(
                str(exc),
                status_code=response.status_code,
                provider=provider,
                model=model,
                attempts=attempt,
            ) from exc

        pricing = self.settings.get_model_pricing(model)
        estimated_cost, cost_unavailable_reason = calculate_cost(usage, pricing)

        logger.info(
            "LLM request succeeded model=%s attempt=%d/%d status=%d latency_ms=%.2f input_tokens=%s output_tokens=%s estimated_cost=%s",
            model,
            attempt,
            max_attempts,
            response.status_code,
            attempt_latency_ms,
            usage.input_tokens,
            usage.output_tokens,
            estimated_cost,
        )

        return LLMResponse(
            text=text,
            model=model,
            provider=provider,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost=estimated_cost,
            pricing_source=pricing.source if pricing else None,
            pricing_last_updated=pricing.last_updated if pricing else None,
            cost_unavailable_reason=cost_unavailable_reason,
            latency_ms=total_latency_ms,
            attempts=attempt,
        )


async def complete(
    messages: Sequence[LLMMessage | Mapping[str, str]],
    model: str | None = None,
    max_tokens: int | None = None,
    *,
    settings: LLMSettings | None = None,
    http_client: httpx.AsyncClient | None = None,
    sleep_fn: SleepFn = asyncio.sleep,
) -> LLMResponse:
    """Send an async completion request through :class:`LLMClient` and normalize the output."""
    client = LLMClient(settings=settings, http_client=http_client, sleep_fn=sleep_fn)
    return await client.complete(messages=messages, model=model, max_tokens=max_tokens)
