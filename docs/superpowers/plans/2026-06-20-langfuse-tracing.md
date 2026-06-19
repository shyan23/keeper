# Self-Hosted Langfuse Tracing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opt-in, self-hosted Langfuse v2 tracing that produces one grouped trace per conversation (conversation → node → LLM call with model/tokens/latency), and is a complete no-op when unconfigured.

**Architecture:** A `tracing.py` factory returns a Langfuse langchain `CallbackHandler` (or `None`). `runtime.cfg()` attaches it to the run config at one chokepoint; LangGraph propagates that config to each node, and the chat wrappers forward the node's `config` into `.invoke()` so LLM calls nest under their node span. Self-host stack ships as a separate `docker-compose.langfuse.yml`.

**Tech Stack:** Langfuse v2 (`langfuse/langfuse:2` + Postgres), `langfuse>=2.0,<3` Python SDK, LangGraph 0.2.60, langchain wrappers (ChatGroq/ChatGoogleGenerativeAI/ChatOllama).

**Spec:** `docs/superpowers/specs/2026-06-20-langfuse-tracing-design.md`

---

## File Structure

- `requirements.txt` (modify) — add `langfuse>=2.0,<3`.
- `app/config.py` (modify) — 3 Langfuse settings.
- `.env.example` (modify/create) — Langfuse vars + privacy note.
- `app/agent/tracing.py` (new) — `tracing_enabled()`, `get_handler()`.
- `app/api/runtime.py` (modify) — `cfg()` attaches the handler.
- `app/agent/llm.py` (modify) — `GroqChat.complete/.structured` forward `config`.
- `app/agent/providers.py` (modify) — `GeminiChat`, `OllamaChat`, `FallbackChat` `.complete/.structured` forward `config`.
- `app/agent/router.py`, `app/agent/nodes/{rag,structured,edit,ingest}.py` (modify) — 9 call sites pass `config=config`.
- `docker-compose.langfuse.yml` (new) — self-host stack.
- `README` (modify) — setup steps.
- `tests/agent/test_tracing.py` (new) — unit tests.

---

### Task 1: Dependency + config settings + env example

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py:42` (end of Settings fields)
- Modify/Create: `.env.example`
- Test: `tests/agent/test_tracing.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_tracing.py`:

```python
from app.config import Settings


def test_langfuse_settings_default_off(monkeypatch):
    # Construct Settings ignoring any real .env so defaults are deterministic.
    monkeypatch.setattr(Settings, "model_config",
                        {"env_file": None, "extra": "ignore"}, raising=True)
    s = Settings(database_url="postgresql://x/y")
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host == "http://localhost:3000"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: FAIL — `AttributeError`/`ValidationError`: `langfuse_public_key` not a field.

- [ ] **Step 3: Add the settings**

In `app/config.py`, after the `app_version` line (currently line 42), add:

```python
    # Self-hosted Langfuse tracing (opt-in). Tracing is OFF unless BOTH keys are
    # set. host MUST stay local — never cloud.langfuse.com (medical prompts).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "http://localhost:3000"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: PASS.

- [ ] **Step 5: Add the dependency and env example**

In `requirements.txt`, add a new line at the end:
```
langfuse>=2.0,<3
```
Install: `.venv/bin/pip install "langfuse>=2.0,<3"`
Verify: `.venv/bin/python -c "from langfuse.callback import CallbackHandler; print('ok')"`
Expected: prints `ok`.

Append to `.env.example` (create it if it does not exist):
```
# Self-hosted Langfuse tracing (optional). Leave blank to disable.
# HOST MUST be local — never cloud.langfuse.com; traces contain medical prompts.
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=http://localhost:3000
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/config.py .env.example tests/agent/test_tracing.py
git commit -m "feat(h2): langfuse dependency + tracing settings (off by default)"
```

---

### Task 2: `tracing.py` handler factory

**Files:**
- Create: `app/agent/tracing.py`
- Test: `tests/agent/test_tracing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_tracing.py`:

```python
from app.agent import tracing
from app.config import get_settings


def test_tracing_disabled_when_keys_missing(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_public_key", None, raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", None, raising=True)
    assert tracing.tracing_enabled() is False
    assert tracing.get_handler("thread-1") is None


def test_tracing_enabled_when_both_keys_set(monkeypatch):
    monkeypatch.setattr(get_settings(), "langfuse_public_key", "pk-test", raising=True)
    monkeypatch.setattr(get_settings(), "langfuse_secret_key", "sk-test", raising=True)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    assert tracing.tracing_enabled() is True
    handler = tracing.get_handler("thread-1")
    assert handler is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.tracing'`.

- [ ] **Step 3: Write the implementation**

Create `app/agent/tracing.py`:

```python
from __future__ import annotations

import logging

from app.config import get_settings

logger = logging.getLogger(__name__)


def tracing_enabled() -> bool:
    """True only when both Langfuse keys are configured."""
    s = get_settings()
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def get_handler(session_id: str | None = None):
    """A Langfuse langchain CallbackHandler, or None when tracing is disabled.

    session_id groups a conversation's turns under one Langfuse session. The
    handler reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST from
    the environment (populated from .env).
    """
    if not tracing_enabled():
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(session_id=session_id)
    except Exception:  # noqa: BLE001 - tracing must never break a request
        logger.warning("Langfuse handler unavailable; continuing untraced.",
                       exc_info=True)
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: PASS (all tracing tests).

- [ ] **Step 5: Commit**

```bash
git add app/agent/tracing.py tests/agent/test_tracing.py
git commit -m "feat(h2): tracing.py langfuse handler factory"
```

---

### Task 3: `cfg()` attaches the handler

**Files:**
- Modify: `app/api/runtime.py:46-54` (the `cfg` function)
- Test: `tests/agent/test_tracing.py`

Current `cfg` (for reference):
```python
def cfg(thread_id: str, deps: Any = None,
        progress: Callable[[str], None] | None = None) -> dict:
    configurable: dict[str, Any] = {
        "deps": deps if deps is not None else get_deps(),
        "thread_id": thread_id,
    }
    if progress is not None:
        configurable["progress"] = progress
    return {"configurable": configurable}
```

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_tracing.py`:

```python
from app.api import runtime


def test_cfg_omits_callbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: None)
    c = runtime.cfg("thread-1", deps=object())
    assert "callbacks" not in c


def test_cfg_attaches_callbacks_when_enabled(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(runtime, "get_handler", lambda session_id=None: sentinel)
    c = runtime.cfg("thread-1", deps=object())
    assert c["callbacks"] == [sentinel]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_tracing.py::test_cfg_attaches_callbacks_when_enabled -v`
Expected: FAIL — `AttributeError: module 'app.api.runtime' has no attribute 'get_handler'`.

- [ ] **Step 3: Modify `runtime.py`**

Add to the imports at the top of `app/api/runtime.py`:
```python
from app.agent.tracing import get_handler
```

Replace the `cfg` function body with:
```python
def cfg(thread_id: str, deps: Any = None,
        progress: Callable[[str], None] | None = None) -> dict:
    configurable: dict[str, Any] = {
        "deps": deps if deps is not None else get_deps(),
        "thread_id": thread_id,
    }
    if progress is not None:
        configurable["progress"] = progress
    config: dict[str, Any] = {"configurable": configurable}
    handler = get_handler(session_id=thread_id)
    if handler is not None:
        config["callbacks"] = [handler]
    return config
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/api/runtime.py tests/agent/test_tracing.py
git commit -m "feat(h2): cfg() attaches langfuse handler when enabled"
```

---

### Task 4: Thread `config` through the chat wrappers

Each chat wrapper method gains `config=None` and forwards it to `.invoke`. This
is what makes LLM calls nest under their node span.

**Files:**
- Modify: `app/agent/llm.py:43-74` (`GroqChat.complete`, `GroqChat.structured`)
- Modify: `app/agent/providers.py:28-32, 78-82, 224-242` (`GeminiChat`, `OllamaChat`, `FallbackChat`)
- Test: `tests/agent/test_tracing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_tracing.py`:

```python
from app.agent.llm import GroqChat
from app.agent.providers import FallbackChat


class _RecordingInner:
    """Stands in for a langchain chat client; records the config it was invoked with."""
    def __init__(self):
        self.seen_config = "UNSET"

    def invoke(self, prompt, config=None):
        self.seen_config = config
        class _R:
            content = "ok"
        return _R()


def test_groqchat_complete_forwards_config():
    inner = _RecordingInner()
    chat = GroqChat(inner=inner)
    chat.complete("hi", config={"callbacks": ["h"]})
    assert inner.seen_config == {"callbacks": ["h"]}


class _RecordingProvider:
    def __init__(self):
        self.seen_config = "UNSET"

    def complete(self, prompt, config=None):
        self.seen_config = config
        return "ok"


def test_fallbackchat_complete_forwards_config():
    p = _RecordingProvider()
    chat = FallbackChat([p])
    chat.complete("hi", config={"callbacks": ["h"]})
    assert p.seen_config == {"callbacks": ["h"]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_tracing.py::test_groqchat_complete_forwards_config tests/agent/test_tracing.py::test_fallbackchat_complete_forwards_config -v`
Expected: FAIL — `TypeError: complete() got an unexpected keyword argument 'config'`.

- [ ] **Step 3: Modify `GroqChat` in `app/agent/llm.py`**

Replace `complete` (line 43-44):
```python
    def complete(self, prompt: str, config=None) -> str:
        return self._chat_client().invoke(prompt, config=config).content
```

Replace `structured` (line 46-74) — add `config=None` and forward it in BOTH the strict and json_object paths:
```python
    def structured(self, prompt: str, schema: type[BaseModel], config=None) -> BaseModel:
        from app.agent.schema import to_strict_schema
        from app.agent.structured import validate_and_retry

        model = get_settings().groq_structured_model
        client = self._structured_client()

        if model in self.STRICT_MODELS:
            rf = {"type": "json_schema",
                  "json_schema": {"name": schema.__name__, "strict": True,
                                  "schema": to_strict_schema(
                                      schema, drop={"source_span", "confidence"})}}
            raw = client.bind(response_format=rf).invoke(prompt, config=config).content
            return schema.model_validate_json(raw)

        base = (f"{prompt}\n\nReturn ONLY a JSON object — no prose, no code fences — "
                f"matching this JSON schema:\n{json.dumps(schema.model_json_schema())}")

        def invoke_raw(extra: str) -> str:
            return client.bind(
                response_format={"type": "json_object"}
            ).invoke(base + extra, config=config).content

        return validate_and_retry(invoke_raw, schema)
```
(Keep the existing explanatory comments in that block; only the signature and the two `.invoke(...)` calls change.)

- [ ] **Step 4: Modify `GeminiChat` and `OllamaChat` in `app/agent/providers.py`**

`GeminiChat` (lines 28-32):
```python
    def complete(self, prompt: str, config=None) -> str:
        return self._inner.invoke(prompt, config=config).content

    def structured(self, prompt: str, schema: type[BaseModel], config=None) -> BaseModel:
        return self._inner.with_structured_output(schema).invoke(prompt, config=config)
```

`OllamaChat` (lines 78-82):
```python
    def complete(self, prompt: str, config=None) -> str:
        return self._inner.invoke(prompt, config=config).content

    def structured(self, prompt: str, schema: type[BaseModel], config=None) -> BaseModel:
        return self._inner.with_structured_output(schema).invoke(prompt, config=config)
```

- [ ] **Step 5: Modify `FallbackChat` in `app/agent/providers.py`**

Replace `complete` and `structured` (lines 224-242) to accept and forward `config`:
```python
    def complete(self, prompt: str, config=None) -> str:
        last = None
        for p in self._providers:
            try:
                return p.complete(prompt, config=config)
            except Exception as e:  # noqa: BLE001 - fallback is the whole point
                log.warning("chat provider %s failed: %s", type(p).__name__, e)
                last = e
        raise RuntimeError(f"all chat providers failed: {last}")

    def structured(self, prompt: str, schema: type[BaseModel], config=None) -> BaseModel:
        last = None
        for p in self._providers:
            try:
                return p.structured(prompt, schema, config=config)
            except Exception as e:  # noqa: BLE001
                log.warning("chat(structured) provider %s failed: %s", type(p).__name__, e)
                last = e
        raise RuntimeError(f"all chat providers failed: {last}")
```

- [ ] **Step 6: Run the wrapper tests + full suite**

Run: `.venv/bin/pytest tests/agent/test_tracing.py -v`
Expected: PASS (config-forwarding tests green).

Run: `.venv/bin/pytest -q --ignore=tests/agent/test_ingest_nodes.py`
Expected: no NEW failures vs the known baseline (5 pre-existing failures unrelated to this change). If a previously-passing test now fails, investigate before committing.

- [ ] **Step 7: Commit**

```bash
git add app/agent/llm.py app/agent/providers.py tests/agent/test_tracing.py
git commit -m "feat(h2): chat wrappers forward run config to .invoke for tracing"
```

---

### Task 5: Pass `config` at the 9 node/router call sites

Now that the wrappers accept `config`, forward each node's `config` so LLM calls
nest under their node span. Every listed node already takes `config` as its
second parameter.

**Files:**
- Modify: `app/agent/router.py:66`
- Modify: `app/agent/nodes/rag.py:75, 100, 113, 122, 172`
- Modify: `app/agent/nodes/structured.py:85`
- Modify: `app/agent/nodes/edit.py:49`
- Modify: `app/agent/nodes/ingest.py:98`
- Test: `tests/agent/test_tracing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_tracing.py`:

```python
from app.agent.nodes.rag import generate_answer_node


class _RecordingChat:
    def __init__(self):
        self.seen_config = "UNSET"

    def complete(self, prompt, config=None):
        self.seen_config = config
        return "an answer"


def test_generate_answer_node_forwards_config():
    chat = _RecordingChat()
    deps = type("Deps", (), {"chat": chat})()
    config = {"configurable": {"deps": deps}, "callbacks": ["h"]}
    state = {
        "messages": [{"role": "user", "content": "what is my RBC?"}],
        "retrieved": [{"text": "RBC 5.1", "chunk_id": 1, "doc_type": "lab",
                       "report_date": "2026-01-01"}],
    }
    generate_answer_node(state, config)
    assert chat.seen_config == config
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_tracing.py::test_generate_answer_node_forwards_config -v`
Expected: FAIL — `assert 'UNSET' == {...}` (node calls `complete` without `config`).

- [ ] **Step 3: Update the call sites**

In `app/agent/router.py:66`, change the structured call to pass `config=config`:
```python
    decision: IntentDecision = deps.chat.structured(
        ...,                       # keep existing prompt/schema args
        config=config,
    )
```

In `app/agent/nodes/rag.py`, add `config=config` to each:
- line 75: `hyde = deps.chat.complete(_HYDE_PROMPT.format(q=q), config=config)`
- line 100: `raw = deps.chat.complete(_RERANK_PROMPT.format(q=q, snip=h["text"]), config=config)`
- line 113: `score = _to_score(deps.chat.complete(_GRADE_PROMPT.format(q=_last_user_text(state), snips=snips), config=config))`
- line 122: `rewrite = deps.chat.complete(_CORRECT_PROMPT.format(q=q), config=config)`
- line 172: `body = deps.chat.complete(_ANSWER_PROMPT.format(q=_last_user_text(state), snips=snips), config=config)`

In `app/agent/nodes/structured.py:85`:
```python
    f = deps.chat.structured(_PROMPT.format(text=_last_user_text(state)), _Filters, config=config)
```

In `app/agent/nodes/edit.py:49`:
```python
    plan = deps.chat.structured(_EDIT_PROMPT.format(text=_last_user_text(state)),
                                ...,          # keep existing schema arg
                                config=config)
```

In `app/agent/nodes/ingest.py:98` — the call is inside a `lambda`; the lambda must close over `config`:
```python
        lambda: deps.chat.structured(
            ...,                              # keep existing prompt/schema args
            config=config,
        ),
```
(`config` is already a parameter of the enclosing node, so the lambda can reference it.)

- [ ] **Step 4: Run the node test to verify it passes**

Run: `.venv/bin/pytest tests/agent/test_tracing.py::test_generate_answer_node_forwards_config -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/pytest -q --ignore=tests/agent/test_ingest_nodes.py`
Expected: no NEW failures vs the 5-failure baseline. The agent graph tests still pass (config is now threaded but inner clients accept it).

- [ ] **Step 6: Commit**

```bash
git add app/agent/router.py app/agent/nodes/rag.py app/agent/nodes/structured.py app/agent/nodes/edit.py app/agent/nodes/ingest.py tests/agent/test_tracing.py
git commit -m "feat(h2): node call sites forward config for nested LLM spans"
```

---

### Task 6: Self-host compose + README setup

**Files:**
- Create: `docker-compose.langfuse.yml`
- Modify: `README` (the repo's main README; if there are several, the top-level one)

- [ ] **Step 1: Create the compose file**

Create `docker-compose.langfuse.yml`:
```yaml
# Self-hosted Langfuse v2 (lightweight: web + its own Postgres). Local only.
# Start:  docker compose -f docker-compose.langfuse.yml up -d
# UI:     http://localhost:3000  (create a project, copy keys into .env)
services:
  langfuse-server:
    image: langfuse/langfuse:2
    depends_on:
      langfuse-db:
        condition: service_healthy
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@langfuse-db:5432/postgres
      NEXTAUTH_URL: http://localhost:3000
      NEXTAUTH_SECRET: change-me-langfuse-secret
      SALT: change-me-langfuse-salt
      TELEMETRY_ENABLED: "false"
    restart: unless-stopped

  langfuse-db:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    volumes:
      - langfuse_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 3s
      timeout: 3s
      retries: 10
    restart: unless-stopped

volumes:
  langfuse_pgdata:
```

- [ ] **Step 2: Validate the compose file**

Run: `docker compose -f docker-compose.langfuse.yml config -q`
Expected: no output, exit 0 (valid YAML + schema). If `docker compose` is unavailable, run `.venv/bin/python -c "import yaml; yaml.safe_load(open('docker-compose.langfuse.yml')); print('ok')"` instead.

- [ ] **Step 3: Add a README section**

Add a "Tracing (optional, self-hosted Langfuse)" section to the README documenting:
```markdown
## Tracing (optional, self-hosted Langfuse)

Per-conversation traces (node latency + LLM model/tokens), fully local.

1. `docker compose -f docker-compose.langfuse.yml up -d`
2. Open http://localhost:3000, create an account + project.
3. Copy the project's public/secret keys into `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-...
   LANGFUSE_SECRET_KEY=sk-...
   LANGFUSE_HOST=http://localhost:3000
   ```
4. Restart the app. Traces appear per conversation (grouped by thread/session).

Tracing is OFF when the keys are blank. **Keep `LANGFUSE_HOST` local — never
`cloud.langfuse.com`; traces contain medical content.**
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.langfuse.yml README*
git commit -m "feat(h2): self-hosted langfuse compose + setup docs"
```

---

## Self-Review notes

- **Spec coverage:** self-host stack (Task 6), dependency (Task 1), config (Task 1), `.env.example` (Task 1), `tracing.py` factory (Task 2), `cfg()` attach point (Task 3), chat-wrapper config threading (Task 4), node/router call sites (Task 5), README (Task 6), tests (Tasks 1-5). Vision/OCR nesting is explicitly out of scope per the spec — no task, by design.
- **Type/signature consistency:** every wrapper method signature is `(..., config=None)`; `FallbackChat` forwards to `p.complete(prompt, config=config)` / `p.structured(prompt, schema, config=config)`, matching the provider signatures defined in Task 4. `get_handler(session_id=...)` matches its use in `cfg()` (Task 3) and the monkeypatch in tests.
- **No new failures gate:** Tasks 4 and 5 run the full suite (minus the pre-existing `test_ingest_nodes.py` collection error) and require no regressions against the known 5-failure baseline.
- **No-op safety:** with keys blank, `get_handler` returns `None`, `cfg` adds no callbacks, wrappers receive `config=None` → identical behavior to today.
