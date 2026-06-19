# Constrained Structured Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ChatLLM.structured()` reliable by routing Groq structured output to `openai/gpt-oss-120b` with strict `json_schema` constrained decoding, plus a provider-independent validate-and-retry net for json_object paths.

**Architecture:** New schema sanitizer converts a Pydantic model to the OpenAI strict-schema subset. New `validate_and_retry` helper is provider-agnostic. `GroqChat` gets a lazily-built second inner client on the structured model; `structured()` uses strict json_schema when the model supports it, else json_object + retry. No node/caller changes — pure DI. Existing `FallbackChat` still advances providers on error. Eval harness (`make eval`) gates the merge.

**Tech Stack:** Python 3.12, Pydantic v2, langchain-groq 0.2.2 (raw `.bind(response_format=...)`), pytest.

Spec: `.claude/specs/2026-06-19-constrained-structured-decoding-design.md`

---

### Task 1: Config — structured model field

**Files:**
- Modify: `app/config.py:18`

- [ ] **Step 1: Add the setting**

In `app/config.py`, after the `groq_model` line (line 18), add:

```python
    # Structured-output model: strict json_schema (constrained decoding) is only
    # available on Groq for the gpt-oss family — llama-3.3 does json_object only.
    # Free tier: 30 RPM / 8000 TPM. Used only by structured(); complete() stays on groq_model.
    groq_structured_model: str = "openai/gpt-oss-120b"
```

- [ ] **Step 2: Verify it loads**

Run: `python -c "from app.config import get_settings; print(get_settings().groq_structured_model)"`
Expected: `openai/gpt-oss-120b`

- [ ] **Step 3: Commit**

```bash
git add app/config.py
git commit -m "feat(config): add groq_structured_model for constrained decoding"
```

---

### Task 2: Strict-schema sanitizer

**Files:**
- Create: `app/agent/schema.py`
- Test: `tests/agent/test_strict_schema.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_strict_schema.py`:

```python
from app.agent.schema import to_strict_schema
from app.agent.state import ExtractionResult


def test_top_level_object_is_strict():
    s = to_strict_schema(ExtractionResult)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"].keys())


def test_no_defaults_or_titles_remain():
    s = to_strict_schema(ExtractionResult)

    def walk(node):
        if isinstance(node, dict):
            assert "default" not in node
            assert "title" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)


def test_nested_defs_are_strictified():
    # ExtractedTest / ExtractedEntity live under $defs and must also be strict.
    s = to_strict_schema(ExtractionResult)
    assert "$defs" in s
    for sub in s["$defs"].values():
        if sub.get("type") == "object":
            assert sub["additionalProperties"] is False
            assert set(sub["required"]) == set(sub["properties"].keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_strict_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.schema'`

- [ ] **Step 3: Write minimal implementation**

Create `app/agent/schema.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


def to_strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema -> OpenAI strict-mode subset (Groq json_schema strict:true).

    Strict mode requires every object to list ALL properties in `required`, set
    `additionalProperties: false`, and omit `default`. Nested models live under
    `$defs`; `$ref` is preserved (strict mode resolves it)."""
    schema = model.model_json_schema()
    _strictify(schema)
    return schema


def _strictify(node) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        node.pop("default", None)
        node.pop("title", None)
        for v in node.values():
            _strictify(v)
    elif isinstance(node, list):
        for v in node:
            _strictify(v)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_strict_schema.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/schema.py tests/agent/test_strict_schema.py
git commit -m "feat(agent): strict-schema sanitizer for Groq json_schema mode"
```

---

### Task 3: Provider-independent validate-and-retry

**Files:**
- Create: `app/agent/structured.py`
- Test: `tests/agent/test_validate_and_retry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_validate_and_retry.py`:

```python
import pytest
from pydantic import BaseModel
from app.agent.structured import validate_and_retry


class _Schema(BaseModel):
    answer: str


def test_returns_validated_model_first_try():
    out = validate_and_retry(lambda extra: '{"answer": "ok"}', _Schema)
    assert isinstance(out, _Schema) and out.answer == "ok"


def test_retries_then_succeeds_passing_error_back():
    seen = []

    def invoke_raw(extra: str) -> str:
        seen.append(extra)
        return "not json" if len(seen) == 1 else '{"answer": "fixed"}'

    out = validate_and_retry(invoke_raw, _Schema)
    assert out.answer == "fixed"
    assert len(seen) == 2
    assert seen[0] == ""              # first call: no error context
    assert "INVALID" in seen[1]       # retry call: error fed back


def test_raises_after_exhausting_attempts():
    with pytest.raises(Exception):
        validate_and_retry(lambda extra: "still not json", _Schema, attempts=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_validate_and_retry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.structured'`

- [ ] **Step 3: Write minimal implementation**

Create `app/agent/structured.py`:

```python
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ValidationError


def validate_and_retry(invoke_raw: Callable[[str], str],
                       schema: type[BaseModel],
                       attempts: int = 3) -> BaseModel:
    """Parse+validate raw JSON; on ValidationError, re-call invoke_raw with the
    error appended so the model can self-correct. Bounded. Provider-independent.

    invoke_raw(extra_instruction) -> raw JSON string. `extra_instruction` is ""
    on the first call, then the validation error on retries.
    """
    last: Exception | None = None
    extra = ""
    for _ in range(attempts):
        raw = invoke_raw(extra)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last = e
            extra = (f"\n\nYour previous output was INVALID:\n{e}\n"
                     f"Return corrected JSON matching the schema exactly — no prose.")
    raise last  # type: ignore[misc]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_validate_and_retry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/structured.py tests/agent/test_validate_and_retry.py
git commit -m "feat(agent): provider-independent validate_and_retry helper"
```

---

### Task 4: Rewire `GroqChat.structured` (strict + lazy + json_object fallback)

**Files:**
- Modify: `app/agent/llm.py:11-34`
- Test: `tests/agent/test_llm.py` (update existing + add)

- [ ] **Step 1: Update the tests (failing)**

Replace the body of `tests/agent/test_llm.py` with:

```python
import pytest
from pydantic import BaseModel
from app.agent.llm import GroqChat


class _Schema(BaseModel):
    answer: str


class _R:
    def __init__(self, content):
        self.content = content


class _FakeLC:
    """Fake langchain client. Records the response_format bound by structured()."""
    def __init__(self, payload='{"answer": "grounded"}'):
        self.payload = payload
        self.bound_format = None

    def invoke(self, prompt):
        return _R("hello world")

    def bind(self, **kwargs):
        self.bound_format = kwargs.get("response_format")
        payload = self.payload

        class _Bound:
            def invoke(_self, prompt):
                return _R(payload)
        return _Bound()


def test_chat_complete_uses_chat_client():
    chat = GroqChat(inner=_FakeLC())
    assert chat.complete("hi") == "hello world"


def test_structured_strict_path_uses_json_schema(monkeypatch):
    # llm.py does `from app.config import get_settings`, so patch the name bound
    # in the llm module, not app.config.
    monkeypatch.setattr("app.agent.llm.get_settings",
                        lambda: _settings("openai/gpt-oss-120b"))
    fake = _FakeLC()
    chat = GroqChat(structured_inner=fake)
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema) and out.answer == "grounded"
    assert fake.bound_format["type"] == "json_schema"
    assert fake.bound_format["json_schema"]["strict"] is True
    assert fake.bound_format["json_schema"]["name"] == "_Schema"


def test_structured_nonstrict_path_uses_json_object(monkeypatch):
    monkeypatch.setattr("app.agent.llm.get_settings",
                        lambda: _settings("llama-3.3-70b-versatile"))
    fake = _FakeLC()
    chat = GroqChat(structured_inner=fake)
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema) and out.answer == "grounded"
    assert fake.bound_format["type"] == "json_object"


def test_structured_inner_is_lazy():
    # complete() must not build the structured (gpt-oss) client.
    chat = GroqChat(inner=_FakeLC())
    chat.complete("hi")
    assert chat._structured_inner is None


def _settings(structured_model):
    class S:
        groq_model = "llama-3.3-70b-versatile"
        groq_structured_model = structured_model
        groq_api_key = "x"
    return S()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_llm.py -v`
Expected: FAIL — `GroqChat()` has no `structured_inner` param / no `_structured_inner` attr.

- [ ] **Step 3: Rewrite GroqChat**

Replace the `GroqChat` class (`app/agent/llm.py:11-34`) with:

```python
class GroqChat:
    # Strict json_schema (constrained decoding) is only available on these Groq
    # models; everything else falls back to best-effort json_object + retry.
    STRICT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

    def __init__(self, inner=None, structured_inner=None):
        # Both clients are lazy: complete()-only callers never build the gpt-oss
        # structured client, and vice-versa.
        self._inner = inner
        self._structured_inner = structured_inner

    def _chat_client(self):
        if self._inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            self._inner = ChatGroq(model=s.groq_model, api_key=s.groq_api_key,
                                   temperature=0)
        return self._inner

    def _structured_client(self):
        if self._structured_inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            self._structured_inner = ChatGroq(model=s.groq_structured_model,
                                              api_key=s.groq_api_key, temperature=0)
        return self._structured_inner

    def complete(self, prompt: str) -> str:
        return self._chat_client().invoke(prompt).content

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        from app.agent.schema import to_strict_schema
        from app.agent.structured import validate_and_retry

        model = get_settings().groq_structured_model
        client = self._structured_client()

        if model in self.STRICT_MODELS:
            # Constrained decoding: tokens forced to the schema, no truncation.
            rf = {"type": "json_schema",
                  "json_schema": {"name": schema.__name__, "strict": True,
                                  "schema": to_strict_schema(schema)}}
            raw = client.bind(response_format=rf).invoke(prompt).content
            return schema.model_validate_json(raw)

        # Best-effort json_object (e.g. llama-3.3) + self-correcting retry.
        base = (f"{prompt}\n\nReturn ONLY a JSON object — no prose, no code fences — "
                f"matching this JSON schema:\n{json.dumps(schema.model_json_schema())}")

        def invoke_raw(extra: str) -> str:
            return client.bind(
                response_format={"type": "json_object"}
            ).invoke(base + extra).content

        return validate_and_retry(invoke_raw, schema)
```

(`json` and `get_settings` are already imported at the top of `llm.py`.)

- [ ] **Step 4: Run the full agent llm/provider suite**

Run: `pytest tests/agent/test_llm.py tests/agent/test_providers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/llm.py tests/agent/test_llm.py
git commit -m "feat(agent): GroqChat.structured uses gpt-oss-120b strict json_schema"
```

---

### Task 5: Full suite + eval no-regression gate

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -q`
Expected: all pass (no regressions in the 173-test suite). Fix any breakage before continuing.

- [ ] **Step 2: Capture the eval baseline (pre-change reference)**

The current scorecard lives in `eval/last_run.json` from the prior run. Record its
metrics (scalar_accuracy, test_value_recall, entity_recall) as the baseline. If stale,
regenerate on `feat/pdf-generation` before this branch's change for a clean baseline.

- [ ] **Step 3: Run eval on the new code**

Run: `GROQ_API_KEY=$GROQ_API_KEY make eval`
Expected: scorecard prints. Requires a Groq key + network. (gpt-oss-120b free tier:
30 RPM / 8000 TPM — extraction cases run serially, well within limits.)

- [ ] **Step 4: Compare and gate**

Confirm `scalar_accuracy`, `test_value_recall`, `entity_recall` are **≥ baseline** and
`errors` did not increase. If any metric regresses, do not merge — investigate (likely a
strict-schema rejection or a prompt mismatch with gpt-oss) before proceeding.

- [ ] **Step 5: Finish the branch**

Use the superpowers:finishing-a-development-branch skill to decide merge/PR.

---

## Notes / out of scope
- Per-node model routing (roadmap #2) is the next slice and stacks on this.
- Wrapping `OllamaChat`/`GeminiChat` in `validate_and_retry` is deferred.
- TPM backoff for bulk ingest: note only; revisit if batch ingest hits 8000 TPM.
