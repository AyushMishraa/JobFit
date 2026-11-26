# JobFit Core — Handwritten LLM Client (Google Gemini)

A lightweight, reusable asynchronous Python HTTP client configured for **Google Gemini** (`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`) without framework magic. Built directly on `httpx.AsyncClient`, `pydantic`, and `pydantic-settings` so HTTP transport, timeouts, retries, logging, token accounting, and cost estimation are transparent and easy to reason about. Provider request/response formatting is isolated via `_PROVIDERS` (supporting `gemini` by default, as well as `openai` and `anthropic`).

## Why This Exists

Production LLM calls fail regularly due to rate limits (`429`), transient provider outages (`500`, `502`, `503`, `529`), and stalled network connections. Putting retry, timeout, logging, and cost-tracking logic in one shared client prevents every downstream feature from duplicating fragile API code.

## Project Structure

```text
JobFit Project/
├── app/
│   ├── clients/
│   │   ├── __init__.py
│   │   └── llm_client.py      # Async HTTP client, retries, backoff, logging, cost math
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py        # Env settings (GEMINI_API_KEY/LLM_API_KEY), timeouts, pricing
│   └── schemas/
│       ├── __init__.py
│       └── llm.py             # LLMMessage, TokenUsage, ModelPricing, LLMResponse
├── tests/
│   ├── conftest.py            # Deterministic test settings & fixtures
│   └── test_llm_client.py     # Mocked HTTP tests via respx (zero real API calls)
├── .env.example               # Safe environment variable template
├── .gitignore
└── pyproject.toml
```

## Setup & Installation

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install runtime and test dependencies:

```powershell
pip install -e ".[dev]"
```

3. Configure environment variables:

```powershell
Copy-Item .env.example .env
# Edit .env and set GEMINI_API_KEY (or GOOGLE_API_KEY / LLM_API_KEY)
```

## Usage

Call `complete(...)` from `app.clients.llm_client`:

```python
import asyncio
from app.clients.llm_client import complete


async def main() -> None:
    response = await complete(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize why retries need jitter."},
        ],
        model="gemini-2.5-flash",
        max_tokens=256,
    )
    print(response.text)
    print(f"Tokens: in={response.input_tokens}, out={response.output_tokens}")
    print(f"Estimated cost (USD): {response.estimated_cost}")


asyncio.run(main())
```

Or run the built-in module demo directly:

```powershell
.\.venv\Scripts\python.exe -m app.clients.llm_client
```

## Normalized `LLMResponse` Shape

Every successful call returns an `LLMResponse` Pydantic model (`app/schemas/llm.py`):

| Field | Type | Description |
| --- | --- | --- |
| `text` | `str` | Generated assistant text |
| `model` | `str` | Model name used for the request (e.g. `gemini-2.5-flash`) |
| `provider` | `str` | Active provider (`"gemini"` by default, or `"openai"` / `"anthropic"`) |
| `input_tokens` | `int \| None` | Prompt/input token count |
| `output_tokens` | `int \| None` | Output token count (including Gemini `thoughtsTokenCount` when present) |
| `total_tokens` | `int \| None` | Total tokens (provider-supplied or `input + output`) |
| `estimated_cost` | `float \| None` | Estimated USD cost, or `None` if unavailable |
| `currency` | `str` | Always `"USD"` |
| `pricing_source` | `str \| None` | Source attribution for the model's rates |
| `pricing_last_updated` | `str \| None` | ISO date when pricing was last updated |
| `cost_unavailable_reason` | `str \| None` | Explicit explanation if `estimated_cost` is `None` |
| `latency_ms` | `float` | End-to-end duration in milliseconds |
| `attempts` | `int` | Number of HTTP attempts made (`1` on first-try success) |

## Retry Policy, Backoff, Jitter, and `Retry-After`

### Retryable vs. Permanent Failures

- **Retried (temporary failures):**
  - HTTP `429` (rate limit), `500` (internal server error), `502` (bad gateway), `503` (service unavailable), `529` (provider overloaded)
  - Transient `httpx` errors: `TimeoutException` (`ConnectTimeout`, `ReadTimeout`, `WriteTimeout`, `PoolTimeout`), `NetworkError`, and `RemoteProtocolError`
  - Bounded by `LLM_MAX_RETRIES` (default `4`, meaning up to `5` total attempts). Exhausting retries raises `LLMRetryExhaustedError`.
- **Fail-fast (permanent errors — never retried):**
  - HTTP `400` (bad request), `401` (unauthorized), `403`, `404`, `422`, and any other non-retryable status code immediately raise `LLMPermanentError`.
  - Malformed JSON or unexpected provider payloads raise `LLMResponseParseError`.
  - Missing `GEMINI_API_KEY` / `GOOGLE_API_KEY` / `LLM_API_KEY` or empty messages raise `LLMConfigurationError` before any network call.

### Backoff Schedule, Jitter, and `Retry-After`

1. If a retryable response includes a valid `Retry-After` header (either non-negative seconds like `"3"` or an HTTP-date), the client waits that duration, clamped to `LLM_BACKOFF_MAX_SECONDS` (default `16.0`s).
2. Otherwise, the client uses exponential backoff:
   - Attempt 1 failure -> base `1s`
   - Attempt 2 failure -> base `2s`
   - Attempt 3 failure -> base `4s`
   - Attempt 4 failure -> base `8s`
   - General formula: `min(backoff_base * 2^(attempt - 1), backoff_max) + random.uniform(0, jitter_max)`, capped at `backoff_max`.
3. Jitter prevents synchronized "thundering herd" retry spikes across concurrent clients.

## Timeouts

Explicit `httpx.Timeout` bounds are configured via environment settings (`app/config/settings.py`):

- `LLM_CONNECT_TIMEOUT` (default `5.0`s): establishing the TCP/TLS connection
- `LLM_READ_TIMEOUT` (default `30.0`s): waiting for response bytes from the server
- `LLM_WRITE_TIMEOUT` (default `10.0`s): sending request body bytes
- `LLM_POOL_TIMEOUT` (default `5.0`s): acquiring a connection from the pool

When `complete()` manages its own `httpx.AsyncClient`, it uses `async with httpx.AsyncClient(timeout=timeout)` so sockets and connections are always closed cleanly even if a request times out.

## Token Usage and Cost Estimation

Gemini returns `usageMetadata.promptTokenCount`, `candidatesTokenCount` (plus `thoughtsTokenCount` on thinking models), and `totalTokenCount`. Estimated cost is calculated from centralized pricing in `DEFAULT_MODEL_PRICING` (`app/config/settings.py`) for `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-2.5-pro`, and `gemini-2.0-flash`:

```text
estimated_cost =
    (input_tokens / 1_000_000 * input_price_per_million)
    + (output_tokens / 1_000_000 * output_price_per_million)
```

If either token usage or model pricing is missing, `estimated_cost` is explicitly `None` and `cost_unavailable_reason` explains why, rather than reporting a misleading `$0.00`.

## Security & Logging

The Gemini API key is sent via the `x-goog-api-key` header (not in the URL query string). Each attempt logs `model`, `attempt`, `status` (when available), `latency_ms`, and retry `delay_s`/`reason`. API keys (`SecretStr`), auth headers, and prompt message contents are never logged, and provider error messages are scrubbed before raising.

## Running Tests

All tests use `respx` to mock HTTP transport and never call a real LLM API:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
