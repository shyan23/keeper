# Confidence-gated conversational router

**Date:** 2026-06-19
**Branch:** feat/pdf-generation (follow-on)
**Status:** Approved design

## Goal

Make the chatbot conversationally intelligent across turns:

- **Ask a clarifying question** when intent confidence is **< 0.80** instead of guessing.
- **Trigger a human-in-the-loop confirm** when intent confidence is **0.80–0.90** before running any chain.
- **Proceed normally** at **≥ 0.90**.

Confidence is LLM self-rated, sourced from a single structured router call.

## Current state

- `app/agent/router.py::classify_intent` returns a bare label (no confidence) from `deps.chat.complete`, reading only the **last** user message.
- `app/agent/graph.py` routes that label to per-intent chains. `ingest`, `edit`, `generate_pdf` already have their own HITL confirm gates; `structured_query` and `rag_query` run without a router-level gate (rag has its own low-confidence gate).
- `MemorySaver` checkpointer already persists `messages` per `thread_id`, so multi-turn history exists but the router ignores it.
- `deps.chat.structured(prompt, schema)` exists (returns a parsed Pydantic model) — used by extraction/report.

## Design

### 1. Confidence-scored router

New schema (in `app/agent/state.py`, next to the other Pydantic schemas):

```python
class IntentDecision(BaseModel):
    intent: Literal["ingest", "structured_query", "rag_query", "edit", "generate_pdf"]
    confidence: float = 0.5          # 0..1, LLM self-rated
    question: str | None = None      # clarifying question, required when ambiguous
```

`classify_intent` rewrite:

1. `file_path` present → `{"intent": "ingest"}` (unchanged, implicit confidence 1.0).
2. Else call `deps.chat.structured(prompt, IntentDecision)` over the **recent conversation** (last ~6 messages via a new `_recent_conversation(state, n=6)` helper), so a reply to a clarifying question fuses with the original request.
3. Branch on `confidence` using module constants `CLARIFY_BELOW = 0.80`, `CONFIRM_BELOW = 0.90`:
   - `>= CONFIRM_BELOW` → `{"intent": <intent>, "route_gate": "go"}`
   - `CLARIFY_BELOW <= c < CONFIRM_BELOW` → `{"intent": <intent>, "route_gate": "confirm"}`
   - `c < CLARIFY_BELOW` → emit the clarifying question via `_say(state, question)` and `{"intent": "clarify", "route_gate": "clarify"}` (turn ends).

`question` fallback: if the LLM returns low confidence but no question, synthesize a generic one ("Could you rephrase — did you want me to look something up, change a value, or make a report?").

### 2. Multi-turn memory

No new storage. `_recent_conversation` joins the last N messages (role-tagged) into the router prompt. The checkpointer already carries history across turns; the only change is the router consuming it.

### 3. Graph wiring

New node `confirm_intent_node` (in router.py or a small new module). It `interrupt(...)`s with the candidate intent, then maps approve→continue / reject→END.

```
START → router
router ──clarify──▶ END
router ──confirm──▶ confirm_intent ──approve──▶ │intent map│
                                   ──reject───▶ END
router ──go───────▶ │intent map│
```

- `_route_gate(state)` reads `state["route_gate"]` → `"clarify" | "confirm" | "go"`.
- The existing intent→entry-node map (`ingest→dedup_check`, `structured_query→parse_filters`, `rag_query→require_patient`, `edit→plan_edit`, `generate_pdf→plan_report`) is reused by **both** the `go` edge and `confirm_intent`'s approve edge. Extract it to a module-level `INTENT_ENTRY` dict to avoid duplication.
- `confirm_intent` interrupt payload: `{"type": "confirm_intent", "intent": <intent>, "summary": <one-line human description>}`. Resume payload: `{"approve": bool}`.

### 4. State additions

`AgentState` (TypedDict): add `route_gate: str | None`. `intent` already exists; `clarify` becomes a valid transient value routed to END.

### 5. UI

- **Clarify**: plain assistant bubble — already rendered via `textContent` (main.ts:107). No new UI.
- **confirm_intent card**: one new branch in the interrupt renderer in `medagentic-dashboard/src/main.ts`, modeled on the `confirm_report` card (Approve / Cancel buttons). Approve sends `{approve: true}`, Cancel sends `{approve: false}` to the existing `/api/chat/resume` flow.

## Testing

- **Router unit** (`tests/agent/test_router.py`): mock `deps.chat.structured` to return high / mid / low confidence `IntentDecision`s → assert `route_gate` is `go` / `confirm` / `clarify` and that clarify appends an assistant message.
- **Graph** (`tests/agent/`): low-confidence run ends with a clarifying answer and no chain side effects; mid-confidence run interrupts with `type == "confirm_intent"`; approve resume continues to the chain.

## Known v1 limitation

Chains parse their own request from the **last** user message (`_last_user_text`), not the fused conversation. After a clarify round the fused context lives in history but the chain re-parses only the latest line. The chains' existing confirm gates (ingest/edit/report) let the user correct this, so it is acceptable for v1. A later pass can thread the fused query into chain parsing.

## Out of scope

- Per-action confidence inside `structured_query` / `rag` nodes (router-level gating only).
- Button-style clarify cards (plain-text clarify only).
- Persisting confidence history / analytics.
