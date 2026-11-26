"""Environment-based configuration and model pricing for the LLM client."""

from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.schemas.llm import ModelPricing

# Centralized model pricing table (USD per 1M tokens).
# Update `last_updated` and rates when provider pricing changes.
DEFAULT_MODEL_PRICING: dict[str, ModelPricing] = {
    "mistral-small-latest": ModelPricing(
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "mistral-large-latest": ModelPricing(
        input_price_per_million=0.70,
        output_price_per_million=2.10,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "codestral-latest": ModelPricing(
        input_price_per_million=0.30,
        output_price_per_million=0.90,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "ministral-3b-latest": ModelPricing(
        input_price_per_million=0.10,
        output_price_per_million=0.10,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "ministral-8b-latest": ModelPricing(
        input_price_per_million=0.15,
        output_price_per_million=0.15,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "ministral-14b-latest": ModelPricing(
        input_price_per_million=0.20,
        output_price_per_million=0.20,
        source="Mistral AI Pricing (https://mistral.ai/pricing)",
        last_updated="2026-11-01",
    ),
    "gpt-4o-mini": ModelPricing(
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        source="OpenAI API Pricing (https://openai.com/api/pricing/)",
        last_updated="2025-05-01",
    ),
    "gpt-4o": ModelPricing(
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        source="OpenAI API Pricing (https://openai.com/api/pricing/)",
        last_updated="2025-05-01",
    ),
    "gpt-4.1-mini": ModelPricing(
        input_price_per_million=0.40,
        output_price_per_million=1.60,
        source="OpenAI API Pricing (https://openai.com/api/pricing/)",
        last_updated="2025-05-01",
    ),
    "gpt-4.1": ModelPricing(
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        source="OpenAI API Pricing (https://openai.com/api/pricing/)",
        last_updated="2025-05-01",
    ),
    "claude-3-5-haiku-latest": ModelPricing(
        input_price_per_million=0.80,
        output_price_per_million=4.00,
        source="Anthropic Pricing (https://www.anthropic.com/pricing)",
        last_updated="2025-05-01",
    ),
    "claude-3-5-sonnet-latest": ModelPricing(
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        source="Anthropic Pricing (https://www.anthropic.com/pricing)",
        last_updated="2025-05-01",
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        input_price_per_million=3.00,
        output_price_per_million=15.00,
        source="Anthropic Pricing (https://www.anthropic.com/pricing)",
        last_updated="2025-05-14",
    ),
}

DEFAULT_PROVIDER_BASE_URLS: dict[str, str] = {
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}


class LLMSettings(BaseSettings):
    """Settings loaded from environment variables and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["mistral", "openai", "anthropic"] = "mistral"
    llm_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("llm_api_key", "LLM_API_KEY", "MISTRAL_API_KEY"),
    )
    llm_base_url: str | None = None
    llm_default_model: str = "mistral-small-latest"
    llm_default_max_tokens: int = Field(default=1024, gt=0)

    # Explicit HTTP timeouts (seconds)
    llm_connect_timeout: float = Field(default=5.0, gt=0.0)
    llm_read_timeout: float = Field(default=30.0, gt=0.0)
    llm_write_timeout: float = Field(default=10.0, gt=0.0)
    llm_pool_timeout: float = Field(default=5.0, gt=0.0)

    # Bounded retry & exponential backoff settings
    llm_max_retries: int = Field(default=4, ge=0)
    llm_backoff_base_seconds: float = Field(default=1.0, gt=0.0)
    llm_backoff_max_seconds: float = Field(default=16.0, gt=0.0)
    llm_jitter_max_seconds: float = Field(default=0.5, ge=0.0)

    # Configurable pricing table
    llm_model_pricing: dict[str, ModelPricing] = Field(
        default_factory=lambda: dict(DEFAULT_MODEL_PRICING)
    )

    @property
    def resolved_base_url(self) -> str:
        """Return the configured base URL or the default URL for the active provider."""
        if self.llm_base_url:
            return self.llm_base_url.rstrip("/")
        return DEFAULT_PROVIDER_BASE_URLS[self.llm_provider]

    def get_model_pricing(self, model: str) -> ModelPricing | None:
        """Look up pricing metadata for a model, returning None if unconfigured."""
        return self.llm_model_pricing.get(model)


def get_settings() -> LLMSettings:
    """Create a fresh settings instance from the current environment."""
    return LLMSettings()
