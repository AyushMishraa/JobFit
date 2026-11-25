"""Shared pytest fixtures for mocked LLM client tests."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.config.settings import LLMSettings


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Standard user prompt fixture with recognizable sensitive text for log checks."""
    return [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "SENSITIVE_PROMPT_SECRET_DO_NOT_LOG"},
    ]


@pytest.fixture
def openai_settings() -> LLMSettings:
    """Deterministic OpenAI-compatible settings for unit tests."""
    return LLMSettings(
        llm_provider="openai",
        llm_api_key=SecretStr("sk-test-secret-key-12345"),
        llm_base_url="https://api.openai.com/v1",
        llm_default_model="gpt-4o-mini",
        llm_default_max_tokens=256,
        llm_connect_timeout=5.0,
        llm_read_timeout=30.0,
        llm_write_timeout=10.0,
        llm_pool_timeout=5.0,
        llm_max_retries=4,
        llm_backoff_base_seconds=1.0,
        llm_backoff_max_seconds=16.0,
        llm_jitter_max_seconds=0.5,
    )


@pytest.fixture
def anthropic_settings() -> LLMSettings:
    """Deterministic Anthropic settings for unit tests."""
    return LLMSettings(
        llm_provider="anthropic",
        llm_api_key=SecretStr("sk-ant-test-secret-98765"),
        llm_base_url="https://api.anthropic.com/v1",
        llm_default_model="claude-3-5-haiku-latest",
        llm_default_max_tokens=256,
        llm_max_retries=2,
        llm_backoff_base_seconds=1.0,
        llm_backoff_max_seconds=16.0,
        llm_jitter_max_seconds=0.5,
    )
