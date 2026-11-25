"""Normalized request, response, usage, and pricing schemas for the LLM client."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """A single chat message in provider-independent format."""

    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    """Normalized token counts returned by an LLM provider."""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class ModelPricing(BaseModel):
    """Per-million-token pricing metadata for a specific model."""

    input_price_per_million: float = Field(..., ge=0.0)
    output_price_per_million: float = Field(..., ge=0.0)
    source: str
    last_updated: str


class LLMResponse(BaseModel):
    """Normalized response returned by :func:`app.clients.llm_client.complete`."""

    text: str
    model: str
    provider: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    estimated_cost: float | None = Field(default=None, ge=0.0)
    currency: str = "USD"
    pricing_source: str | None = None
    pricing_last_updated: str | None = None
    cost_unavailable_reason: str | None = None
    latency_ms: float = Field(..., ge=0.0)
    attempts: int = Field(..., ge=1)
