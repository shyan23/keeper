# Constrained / Structured Decoding for `ChatLLM.structured()`

**Date:** 2026-06-19
**Branch target:** new (`feat/constrained-decoding`)
**Roadmap item:** Horizon 0 #1 (constrained decoding) — see
`.claude/plans/2026-06-18-feature-improvement-roadmap.md`
**Status:** design approved, pending spec review

---

## Problem

`ChatLLM.structured(prompt, schema)` is the agent's keystone — **7 callers**:
extraction (`nodes/ingest.py`), segment split (`services/segment.py`), edit plan
(`nodes/edit.py`), RAG filters (`nodes/structured.py`), report parse
(`services/report.py`), eval (`eval/harness.py`).

Current `GroqChat.structured` (`app/agent/llm.py`) uses Groq **`json_object`** mode +
manual `model_validate_json`. `json_object` is *best-effort JSON* — it does **not**
enforce the schema. Observed failure (recorded in code comment): the tests array was
truncated to 1 of 25 items. Extraction baseline ~84.2% scalar accuracy. Invalid /
truncated JSON is a live defect class.

## Constraint

**$0 / free models only.** No paid APIs.

## Key facts (verified 2026-06-19, Groq docs)

- Groq **strict** `json_schema` (`strict:true`) = true constrained decoding (tokens
  forced to schema). **Only on `openai/gpt-oss-20b`, `openai/gpt-oss-120b`.**
- `llama-3.3-70b-versatile` (current `groq_model`) supports **`json_object` only** — not
  listed on the structured-outputs page at all.
- `openai/gpt-oss-120b` is on the **free tier**: 30 RPM, 8000 TPM, no card. Lower TPM
  than llama-3.3 (~12K) → heavy *batch* ingest throttles; fine for single-user self-host.
- `langchain-groq` 0.2.2 `with_structured_output` has **no `json_schema` method** → strict
  mode must go through raw `.bind(response_format=...)` (same call shape as current code).
- Strict json_schema requires an **OpenAI-subset schema**: every property in `required`,
  `additionalProperties:false` on every object, no `default`. Pydantic
  `model_json_schema()` emits defaults + optional-with-default → needs a sanitizer.

## Decision

Switch the **structured** path to `openai/gpt-oss-120b` + strict json_schema = real
constrained decoding, $0. Keep `llama-3.3-70b` for `complete()`. Universal
validate-and-retry net for any non-strict (json_object) path. Prove no regression with
the existing eval harness before merge.

---

## Design

### 1. Config (`app/config.py`)
Add:
```python
groq_structured_model: str = "openai/gpt-oss-120b"
```
Keep `groq_model = "llama-3.3-70b-versatile"` for `complete()`. Two roles, two models.

### 2. Schema sanitizer (`app/agent/schema.py`, new)
```python
def to_strict_schema(model: type[BaseModel]) -> dict: ...
```
Recurse `model.model_json_schema()`; on every object node:
- `additionalProperties = False`
- `required = list(properties.keys())`  (strict mode requires ALL fields present)
- strip `default`, `title`
- leave `$defs` / `$ref` intact (strict mode resolves them) — covers nested
  `ExtractedTest` / `ExtractedEntity`.

Optional fields (`x | None`) already emit `anyOf:[{...},{"type":"null"}]`, which strict
mode accepts — keep as-is (nullable, but required key present).

No inline `assert` self-check. Correctness lives in `tests/` (item C).

### 3. Provider-independent retry helper (`app/agent/structured.py`, new)
```python
def validate_and_retry(invoke_raw, schema, attempts=3):
    """invoke_raw(extra_instruction: str) -> str  (raw JSON text).
    Parse+validate; on ValidationError re-call with the error appended; bounded."""
```
Used by any json_object path. On exhaustion, raises → `FallbackChat` advances provider.
Strict json_schema path does **not** need it (shape guaranteed) but still validates once.

### 4. `GroqChat` (`app/agent/llm.py`)
- `STRICT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}`
- **Lazy** structured inner (item A): build `ChatGroq(model=groq_structured_model)` on
  first `structured()` call, cache on instance. `complete()`-only paths never build it.
- `structured(prompt, schema)`:
  - if `groq_structured_model in STRICT_MODELS`:
    `response_format={"type":"json_schema","json_schema":{"name":schema.__name__,
    "strict":True,"schema":to_strict_schema(schema)}}` → invoke → `model_validate_json`.
  - else: `validate_and_retry(lambda extra: invoke(json_object, prompt+schema+extra), schema)`.

### 5. Other providers
- `OllamaChat.structured`: keep `with_structured_output` (local, schema-guided). Optionally
  wrap in `validate_and_retry` later — out of scope now.
- `GeminiChat.structured`: keep (`with_structured_output`).
- `FallbackChat`: **unchanged** — advances on any exception. gpt-oss strict fails →
  llama-3.3 → Ollama.

### 6. Defensive coercions kept
`ExtractedTest._stringify` / `ExtractionResult._flatten_scalars` model_validators stay.
Strict mode reduces the need but they're harmless and protect the fallback paths.

---

## Data flow (extraction example, unchanged callers)
```
ingest node → deps.chat.structured(_EXTRACT_PROMPT, ExtractionResult)
  → FallbackChat → GroqChat.structured
      → gpt-oss-120b strict json_schema (constrained)  → ExtractionResult
      (on error) → OllamaChat.structured                → ExtractionResult
```

## Error handling
- Strict path parse failure (should be near-impossible): raises → FallbackChat next provider.
- json_object path: `validate_and_retry` bounded loop, then raise → next provider.
- All providers fail: `FallbackChat` raises `RuntimeError` (existing behavior).

## Testing (TDD — item C; no inline asserts)
`tests/test_strict_schema.py`:
- `to_strict_schema(ExtractionResult)`: every object has `additionalProperties False`,
  `required == all properties`, no `default` key, `$defs` preserved for nested models.

`tests/test_validate_and_retry.py`:
- fake `invoke_raw` returns invalid JSON then valid → returns validated model, called twice.
- always-invalid → raises after `attempts`.

`tests/test_groqchat_structured.py`:
- fake inner; strict model → asserts `response_format.type == "json_schema"`, `strict True`.
- non-strict model setting → routes through `validate_and_retry` (json_object).
- structured inner is **not** built until `structured()` is called (lazy — item A).

**Eval gate (merge blocker):** `make eval` before (baseline) and after; merge only if
`scalar_accuracy`, `test_value_recall`, `entity_recall` ≥ baseline. Baseline in
`eval/last_run.json`.

## Out of scope (deferred)
- Per-node model routing (Horizon 0 #2) — next slice, stacks on this.
- OpenMed NER hybrid (#15).
- Wrapping Ollama/Gemini in `validate_and_retry`.
- TPM throttling/backoff for batch ingest (note only; revisit if bulk ingest hits limits).

## Files touched
- `app/config.py` (1 field)
- `app/agent/schema.py` (new, ~30 lines)
- `app/agent/structured.py` (new, ~20 lines)
- `app/agent/llm.py` (`GroqChat.structured` + lazy inner)
- `tests/test_strict_schema.py`, `tests/test_validate_and_retry.py`,
  `tests/test_groqchat_structured.py` (new)
