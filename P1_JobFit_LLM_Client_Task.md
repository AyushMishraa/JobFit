# P1 --- JobFit Core: Handwritten LLM Client

## Goal

Build a small, reusable Python module that sends requests to an LLM API
over HTTP and returns a normalized response containing:

-   Generated text
-   Input and output token usage
-   Estimated request cost

Every later project should call models through this module rather than
implementing provider calls independently.

**Suggested module:** `llm_client.py`\
**Main interface:**

``` python
async def complete(messages, model, max_tokens) -> LLMResponse:
    ...
```

The implementation should be written by hand. Use an AI assistant to
explain concepts and debug issues, but type the implementation yourself
so you can explain every line in an interview.

------------------------------------------------------------------------

## Why This Matters

LLM applications commonly fail because of rate limits, slow responses,
and unreliable networks. A reliable client handles these conditions
consistently and gives every project a single place to call models.

This project also prepares you to answer practical interview questions
such as:

-   How do you handle an HTTP `429 Too Many Requests` response?
-   Which errors should be retried, and which should fail immediately?
-   How does exponential backoff work?
-   How do you prevent a stalled API request from hanging indefinitely?
-   How do you track token usage and estimate LLM costs?
-   How do you test API behavior without calling a real model provider?

Building this without relying on LangChain makes the underlying HTTP,
retry, timeout, and accounting behavior easier to understand and
explain.

------------------------------------------------------------------------

## Core Requirements

### 1. LLM API request

Create an asynchronous HTTP client that:

-   Accepts a list of messages, a model name, and a maximum output-token
    count.
-   Reads the API key and other configuration from environment-based
    settings.
-   Sends the request using `httpx.AsyncClient`.
-   Parses the provider response into a consistent `LLMResponse` model.
-   Returns generated text, token usage, and estimated cost.

Keep provider-specific request and response parsing isolated so the rest
of the application can use a stable interface.

### 2. Retry handling

Retry temporary failures, including:

-   `429` --- rate limit
-   `500` --- internal server error
-   `502` --- bad gateway
-   `503` --- service unavailable
-   `529` --- provider overloaded, where supported
-   Transient network errors and request timeouts

Use exponential backoff with jitter. A sample schedule is:

  Retry attempt     Base wait
  --------------- -----------
  1                  1 second
  2                 2 seconds
  3                 4 seconds
  4                 8 seconds

Add a small random jitter to avoid many clients retrying at exactly the
same time.

If the response includes a valid `Retry-After` header, respect it
instead of blindly using the local backoff delay. Set a sensible maximum
retry count and maximum wait so requests cannot retry forever.

### 3. Do not retry permanent errors

Fail immediately for errors that are unlikely to succeed if repeated,
such as:

-   `400` --- bad request
-   `401` --- unauthorized / invalid API key

Handle other non-retryable client errors similarly unless the provider's
documentation gives a specific reason to retry them.

Raise a clear, useful exception that preserves the original error
context without exposing secrets such as API keys.

### 4. Timeouts

Set explicit HTTP timeouts. For example, use a 30-second read timeout.

A timeout prevents a request from waiting indefinitely when the provider
or network stops responding. Configure connection and write timeouts as
appropriate too.

### 5. Logging

Log each request attempt with useful operational details:

-   Model name
-   Attempt number
-   HTTP status code, when available
-   Request latency
-   Retry delay and reason, when retrying
-   Final success or failure

Never log API keys, authorization headers, or sensitive prompt content.

### 6. Token usage and cost

Read token usage from the provider response. Normalize at least:

-   Input/prompt tokens
-   Output/completion tokens
-   Total tokens, if supplied or derivable

Calculate estimated cost using configurable model pricing:

``` text
estimated_cost =
    (input_tokens / 1_000_000 * input_price_per_million)
    + (output_tokens / 1_000_000 * output_price_per_million)
```

Keep pricing in configuration rather than scattering hard-coded prices
throughout the client. Record the pricing source or last-updated date
because model prices can change. If usage or pricing is unavailable,
represent that explicitly rather than silently reporting a misleading
cost.

------------------------------------------------------------------------

## Suggested Project Structure

Adapt this to the existing repository, but keep responsibilities
separated.

``` text
jobfit/
├── app/
│   ├── clients/
│   │   ├── __init__.py
│   │   └── llm_client.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   └── schemas/
│       ├── __init__.py
│       └── llm.py
├── tests/
│   ├── conftest.py
│   └── test_llm_client.py
├── .env.example
├── .gitignore
├── README.md
└── pyproject.toml
```

### Responsibilities

  -----------------------------------------------------------------------
  File                                Responsibility
  ----------------------------------- -----------------------------------
  `app/clients/llm_client.py`         HTTP request, timeout,
                                      retry/backoff, response
                                      normalization, logging

  `app/config/settings.py`            Environment variables, provider
                                      configuration, timeout/retry
                                      settings, model pricing

  `app/schemas/llm.py`                Pydantic request/response models
                                      and normalized usage data

  `tests/test_llm_client.py`          Mocked HTTP tests for success and
                                      failure scenarios

  `.env.example`                      Names of required environment
                                      variables with safe placeholder
                                      values

  `README.md`                         Setup, usage, retry behavior,
                                      timeout behavior, and testing
                                      instructions
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## Implementation Plan

### Step 1 --- Learn the Python building blocks

Relate the Python tools to technologies you already know:

  -----------------------------------------------------------------------
  Python                              Familiar equivalent
  ----------------------------------- -----------------------------------
  Type hints                          TypeScript types

  Pydantic models                     NestJS DTOs with validation

  `async` / `await`                   JavaScript asynchronous functions

  `asyncio.run()`                     Entry point that starts and runs an
                                      asyncio event loop

  `httpx.AsyncClient`                 Axios-like HTTP client with async
                                      support

  `pytest`                            Jest/Pytest-style automated test
                                      runner

  `respx` / `pytest-httpx`            HTTP request mocking for tests
  -----------------------------------------------------------------------

Understand the concepts before implementing the client. In particular,
learn how exceptions from `httpx` differ from HTTP error status codes.

### Step 2 --- Implement the basic API call

-   Define the normalized Pydantic response model.
-   Load API configuration from settings and environment variables.
-   Create the basic asynchronous HTTP request.
-   Parse the provider response.
-   Return text, token counts, and estimated cost.

At this stage, verify a successful request using a mocked response or a
controlled development call.

### Step 3 --- Add timeouts

-   Configure explicit connection, read, write, and pool timeouts as
    appropriate.
-   Handle timeout exceptions.
-   Ensure a timed-out request does not leave resources open.

### Step 4 --- Add retry logic

-   Define retryable HTTP status codes and transient network exceptions.
-   Add a maximum attempt count.
-   Implement exponential backoff and jitter.
-   Parse and honor `Retry-After` when present.
-   Fail fast for permanent errors.
-   Log each attempt and retry decision.

### Step 5 --- Write automated tests

Use `pytest` with `respx` or `pytest-httpx` to mock the HTTP layer.
Tests must not call a real LLM API.

Cover all five required scenarios:

1.  **Normal success:** a successful provider response returns text,
    token usage, and cost.
2.  **429 then success:** the client retries after a rate-limit response
    and returns the later successful response.
3.  **Repeated server errors:** three consecutive `500` responses result
    in the expected exception after the configured attempts.
4.  **Timeout then success:** a timeout triggers a retry, and a
    subsequent successful response is returned.
5.  **401 is not retried:** an unauthorized response fails immediately
    and makes only one request.

Also consider tests for `Retry-After`, jitter/backoff bounds, malformed
provider responses, missing usage data, and ensuring secrets are not
logged.

------------------------------------------------------------------------

## Definition of Done

-   [ ] `llm_client.py` exposes the agreed asynchronous `complete(...)`
    interface.
-   [ ] A successful call returns generated text, input tokens, output
    tokens, and estimated cost.
-   [ ] API credentials and configuration are loaded from
    environment-based settings.
-   [ ] Explicit HTTP timeouts are configured.
-   [ ] Temporary HTTP and network failures are retried.
-   [ ] Exponential backoff and jitter are implemented.
-   [ ] `Retry-After` is respected when valid.
-   [ ] Permanent errors such as `400` and `401` fail without retrying.
-   [ ] Each attempt logs model, attempt number, status where available,
    and latency.
-   [ ] Logs do not expose credentials or sensitive prompt content.
-   [ ] All five required mocked test scenarios pass.
-   [ ] Tests do not make real provider API calls.
-   [ ] README explains setup, usage, retry policy, timeout behavior,
    token usage, and cost calculation.
-   [ ] The implementation can be explained line by line in an
    interview.

**Deadline:** Ship by Sunday.

------------------------------------------------------------------------

## README Acceptance Criteria

The README should include:

1.  What the LLM client does and why it exists.
2.  Installation and environment setup.
3.  How to call `complete(...)`.
4.  The shape of the normalized `LLMResponse`.
5.  Which failures are retried and which fail immediately.
6.  How exponential backoff, jitter, and `Retry-After` work.
7.  Timeout configuration and behavior.
8.  How token usage and estimated cost are calculated.
9.  How to run the test suite.

------------------------------------------------------------------------

## Interview Preparation

Be ready to explain these topics in your own words:

-   **HTTP 429:** The provider is rate-limiting requests. Retry after
    the specified `Retry-After` duration when available; otherwise use
    bounded exponential backoff with jitter.
-   **Exponential backoff:** Increase the delay after each failed
    attempt to reduce pressure on a struggling service.
-   **Jitter:** Add randomness to retry delays so many clients do not
    retry simultaneously.
-   **Fail-fast errors:** Do not retry permanent errors such as invalid
    requests or invalid credentials.
-   **Timeouts:** Bound how long the application waits for a response
    and handle timeout exceptions explicitly.
-   **Mocked HTTP tests:** Simulate provider responses and failures so
    tests are deterministic, fast, and independent of API availability.
-   **Token accounting:** Read usage returned by the provider and apply
    the configured input/output token rates to estimate cost.

## Important Engineering Notes

-   Confirm the provider's exact request format, response schema, usage
    fields, retryable statuses, and `Retry-After` behavior from its
    documentation.
-   Avoid assuming all providers use the same response structure or
    error format.
-   Keep retries bounded. Retrying increases latency and can amplify
    load during an outage.
-   Consider idempotency and provider billing behavior before retrying
    requests whose outcome is uncertain after a network failure.
-   Do not retry every exception indiscriminately.
-   Keep the core client small and understandable; add abstractions only
    when they solve a real requirement.
