# Self-Hosted Langfuse Tracing — Design

**Date:** 2026-06-20
**Status:** Approved
**Horizon:** H2 (Durability, insight, observability) — sub-project 3 of 3
**Related:** durable checkpointer (`2026-06-20-durable-checkpointer-design.md`, shipped)

## Problem

The agent has per-node progress labels (`runtime.NODE_LABELS`) but no trace
timeline, no token/cost view, and no way to debug a run after the fact. When a
RAG answer is wrong or slow, there is no record of which node was slow, which
model ran, or how many tokens each LLM call used.

## Goal

Local, privacy-safe observability. One trace tree per conversation:
**conversation → node (latency) → LLM call (model / tokens / latency)**, viewable
in a self-hosted Langfuse UI. Disabled by default; zero overhead and a complete
no-op when unconfigured. Medical prompts never leave the machine.

## Decisions (locked during brainstorming)

- **Backend:** Langfuse OSS, **self-hosted** (not LangSmith cloud — medical
  prompts must stay on-box for the privacy-first positioning).
- **Footprint:** Langfuse **v2** (single web container + its own Postgres),
  chosen over v3 (which additionally needs ClickHouse + Redis + MinIO) for a
  lighter dev-machine footprint. SDK pinned to v2 to match.
- **Granularity:** grouped traces — conversation → node → LLM call.

## Architecture

### 1. Self-host stack — `docker-compose.langfuse.yml` (new)

Separate compose file (not merged into any app compose). Two services:
- `langfuse-server` — `langfuse/langfuse:2`, port `3000:3000`, env:
  `DATABASE_URL` (its own Postgres), `NEXTAUTH_URL=http://localhost:3000`,
  `NEXTAUTH_SECRET`, `SALT`, `TELEMETRY_ENABLED=false`.
- `langfuse-db` — `postgres:16`, named volume for persistence. Distinct from the
  app DB and the test DB; nothing shared.

One-time setup (documented in README): `docker compose -f docker-compose.langfuse.yml up -d`,
open `localhost:3000`, create an account + project, copy the project's public/secret
keys into `.env`.

### 2. Dependency

Add `langfuse>=2.0,<3` to `requirements.txt`. SDK v2 talks to server v2; the
langchain handler import path is `from langfuse.callback import CallbackHandler`.

### 3. Config — `app/config.py`

Add to `Settings`:
- `langfuse_public_key: str | None = None`
- `langfuse_secret_key: str | None = None`
- `langfuse_host: str = "http://localhost:3000"`

Add the three vars to `.env.example` with a comment: **must be a local host —
never `cloud.langfuse.com`** (medical content stays on-box).

### 4. Handler factory — `app/agent/tracing.py` (new)

Single responsibility: decide if tracing is on, and build a handler.

```python
def tracing_enabled() -> bool:
    """True only when both Langfuse keys are configured."""
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)

def get_handler(session_id: str | None = None):
    """A Langfuse langchain CallbackHandler, or None when tracing is disabled.
    session_id groups a conversation's turns under one Langfuse session."""
    if not tracing_enabled():
        return None
    from langfuse.callback import CallbackHandler
    return CallbackHandler(session_id=session_id)
```

The handler reads `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`
from the environment (set from `.env`).

### 5. One attach point — `app/api/runtime.py` `cfg()`

`cfg(thread_id, deps, progress)` already builds the run config dict consumed by
**both** `sse.py` and `routes_chat.py`. It gains: when `get_handler(thread_id)`
returns a handler, add `config["callbacks"] = [handler]` at the **top level** of
the returned dict (peer of `configurable`, not inside it). `session_id` is the
`thread_id`, so all turns of one conversation group into a single Langfuse
session. When disabled, no `callbacks` key is added.

This is the only invoke-path wiring needed: LangGraph propagates this run config
(callbacks + parent-run context) to every node as the `config` argument the node
already receives.

### 6. Grouped nesting — thread `config` into the chat wrappers

LangGraph hands each node the run `config`. For LLM calls to nest under their
node's span (rather than escape as untraced calls), the node must forward that
`config` to the LLM wrapper, which forwards it to `.invoke(prompt, config=config)`.

**Wrapper methods gaining `config=None` (forwarded to the inner `.invoke`):**
- `app/agent/providers.py`: `FallbackChat.complete` / `.structured` (the
  dispatcher `deps.chat`) — forwards `config` to each provider it tries.
- `app/agent/providers.py`: `GeminiChat.complete` / `.structured`,
  `OllamaChat.complete` / `.structured`.
- `app/agent/llm.py`: `GroqChat.complete` / `.structured` (incl. the
  `.bind(response_format=...).invoke(...)` path and the `invoke_raw` closure
  used by `validate_and_retry`).

`with_structured_output(schema).invoke(prompt, config=config)` and
`.bind(...).invoke(prompt, config=config)` both accept the config kwarg.

**Node/router call sites gaining `config=config` (9 total):**
- `app/agent/router.py:66` — `classify_intent` structured call.
- `app/agent/nodes/rag.py` — lines 75, 100, 113, 122, 172 (`complete`).
- `app/agent/nodes/structured.py:85` — filters `structured`.
- `app/agent/nodes/edit.py:49` — edit-plan `structured`.
- `app/agent/nodes/ingest.py:98` — segment-extract `structured` (inside the
  retry lambda; the lambda must close over `config`).

Every one of these nodes already takes `config` as its second parameter, so no
node signature changes are required (verify `classify_intent` during
implementation; add the param if missing).

### 7. Out of scope: vision / OCR nested spans

Vision/OCR LLM calls run inside `app/services/extraction.py`
(`vision.ocr_image(...)`), reached from `extract_text_node` via
`extract_text()` / `extract_pages()` — several layers below the run `config`.
Threading `config` through the entire OCR/extraction pipeline is disproportionate
to the value. **The ingest node still appears as a node span (with latency)** in
the trace; only the per-OCR-call token breakdown is omitted. Revisit only if
OCR-call-level token tracking is later needed.

## Error handling & privacy

- **Unconfigured (default):** `get_handler` returns `None`, `cfg()` adds no
  callbacks, wrappers receive `config=None` and behave exactly as today. No
  dependency on a running Langfuse server.
- **Server down while enabled:** the Langfuse handler buffers and flushes
  best-effort; failures are logged by the SDK and must never raise into the
  request path. (The wrappers do not add try/except around `.invoke` for this —
  the SDK is responsible for swallowing its own transport errors.)
- **Privacy:** `LANGFUSE_HOST` defaults to a local URL; README and `.env.example`
  state plainly that it must stay local. Traces contain medical prompts.

## Testing

- `tracing_enabled()` — False when either key missing; True when both set
  (monkeypatch settings).
- `get_handler()` — returns `None` when disabled; returns a handler when enabled.
- `cfg()` — includes a top-level `callbacks` list when enabled; omits it when
  disabled.
- Wrapper config-forwarding — for `GroqChat.complete` and `.structured`, inject a
  fake inner whose `.invoke` records its `config` kwarg; assert the passed
  `config` reaches `.invoke`. Same shape for `FallbackChat` (asserts it forwards
  to the underlying provider).
- No live-Langfuse integration test (would require a running server); a handler
  construction smoke test under monkeypatched keys is sufficient.
- Full suite must still pass (the 5 pre-existing failures noted in the
  checkpointer work remain out of scope).

## File structure

- `docker-compose.langfuse.yml` (new) — self-host stack.
- `app/agent/tracing.py` (new) — `tracing_enabled()`, `get_handler()`.
- `requirements.txt` (modify) — add `langfuse>=2.0,<3`.
- `app/config.py` (modify) — three settings.
- `app/api/runtime.py` (modify) — `cfg()` attaches the handler.
- `app/agent/llm.py`, `app/agent/providers.py` (modify) — wrappers forward `config`.
- `app/agent/router.py`, `app/agent/nodes/{rag,structured,edit,ingest}.py`
  (modify) — call sites pass `config`.
- `.env.example` + `README` (modify) — Langfuse vars + setup steps.
- `tests/agent/test_tracing.py` (new) — unit tests above.

## Out of scope (other H2 work)

- Typed numeric test results + trend backing (H2 sub-project 2, not yet built).
- Cost dashboards beyond what Langfuse provides out of the box.
- Alerting on latency/cost thresholds.
