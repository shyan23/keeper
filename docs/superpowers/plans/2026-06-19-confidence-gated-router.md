# Confidence-gated Conversational Router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the chatbot ask a clarifying question when intent confidence < 0.80, route through a HITL confirm gate when 0.80–0.90, and proceed at ≥ 0.90 — using recent conversation history so clarifications fuse with the original request.

**Architecture:** Replace the bare-label router with an LLM structured call returning `{intent, confidence, question}`. The router sets a `route_gate` (`go`/`confirm`/`clarify`). The graph branches: `clarify` ends the turn with the question, `confirm` interrupts a new `confirm_intent` gate before dispatching to the intent chain, `go` dispatches directly. A reused `INTENT_ENTRY` map wires both the direct and post-confirm dispatch.

**Tech Stack:** LangGraph, Pydantic, FastAPI SSE, pytest. Frontend: vanilla TS (medagentic-dashboard).

---

### Task 1: Add `IntentDecision` schema + `route_gate` state field

**Files:**
- Modify: `app/agent/state.py` (add schema near other Pydantic models ~line 75; add `route_gate` to `AgentState` TypedDict ~line 120)
- Test: `tests/agent/test_state.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/agent/test_state.py`:

```python
def test_intent_decision_defaults():
    from app.agent.state import IntentDecision
    d = IntentDecision(intent="rag_query")
    assert d.confidence == 0.5
    assert d.question is None

def test_intent_decision_rejects_bad_intent():
    import pytest
    from pydantic import ValidationError
    from app.agent.state import IntentDecision
    with pytest.raises(ValidationError):
        IntentDecision(intent="not_a_real_intent")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_state.py -k intent_decision -v`
Expected: FAIL — `ImportError: cannot import name 'IntentDecision'`

- [ ] **Step 3: Add the schema and state field**

In `app/agent/state.py`, add near the other Pydantic schemas (after `ExtractionResult`):

```python
class IntentDecision(BaseModel):
    intent: Literal["ingest", "structured_query", "rag_query", "edit", "generate_pdf"]
    confidence: float = 0.5          # 0..1, LLM self-rated
    question: str | None = None      # clarifying question, set when ambiguous
```

Ensure `Literal` is imported at the top of the file:

```python
from typing import Any, Literal, Protocol, TypedDict
```

In the `AgentState(TypedDict, total=False)` block, add:

```python
    route_gate: str | None    # "go" | "confirm" | "clarify"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_state.py -k intent_decision -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/state.py tests/agent/test_state.py
git commit -m "feat(router): IntentDecision schema + route_gate state field"
```

---

### Task 2: Confidence-scored router with clarify/confirm/go gating

**Files:**
- Modify: `app/agent/router.py` (full rewrite of the LLM path; keep file-upload shortcut)
- Test: `tests/agent/test_router.py` (rewrite — existing tests use `.complete`, new router uses `.structured`)

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/agent/test_router.py` with:

```python
from app.agent.router import classify_intent
from app.agent.state import IntentDecision


class _FakeChat:
    def __init__(self, decision: IntentDecision):
        self._decision = decision

    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
        return self._decision


def _cfg(chat):
    from app.agent.state import Deps
    deps = Deps(chat=chat, vision=None, embedder=None, session_factory=None)
    return {"configurable": {"deps": deps}}


def test_router_ingest_when_file_present():
    state = {"messages": [{"role": "user", "content": "read this"}], "file_path": "/x.png"}
    out = classify_intent(state, _cfg(_FakeChat(IntentDecision(intent="rag_query"))))
    assert out["intent"] == "ingest"
    assert out["route_gate"] == "go"


def test_router_high_confidence_goes():
    state = {"messages": [{"role": "user", "content": "latest report of Jane"}]}
    out = classify_intent(state, _cfg(_FakeChat(
        IntentDecision(intent="structured_query", confidence=0.95))))
    assert out["intent"] == "structured_query"
    assert out["route_gate"] == "go"


def test_router_mid_confidence_confirms():
    state = {"messages": [{"role": "user", "content": "show jane"}]}
    out = classify_intent(state, _cfg(_FakeChat(
        IntentDecision(intent="structured_query", confidence=0.85))))
    assert out["route_gate"] == "confirm"
    assert out["intent"] == "structured_query"


def test_router_low_confidence_clarifies():
    state = {"messages": [{"role": "user", "content": "do the thing"}]}
    out = classify_intent(state, _cfg(_FakeChat(
        IntentDecision(intent="rag_query", confidence=0.3, question="What would you like?"))))
    assert out["route_gate"] == "clarify"
    assert out["intent"] == "clarify"
    assert out["messages"][-1]["role"] == "assistant"
    assert out["messages"][-1]["content"] == "What would you like?"


def test_router_low_confidence_synthesizes_question_when_missing():
    state = {"messages": [{"role": "user", "content": "??"}]}
    out = classify_intent(state, _cfg(_FakeChat(
        IntentDecision(intent="rag_query", confidence=0.1, question=None))))
    assert out["route_gate"] == "clarify"
    assert out["messages"][-1]["content"]  # non-empty fallback question
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/agent/test_router.py -v`
Expected: FAIL — router still returns only `{"intent": ...}`, no `route_gate`; `KeyError`/assertion errors.

- [ ] **Step 3: Rewrite the router**

Replace the contents of `app/agent/router.py` with:

```python
from __future__ import annotations

from typing import Any

from app.agent.state import IntentDecision

CLARIFY_BELOW = 0.80
CONFIRM_BELOW = 0.90

_FALLBACK_QUESTION = (
    "I'm not sure what you'd like me to do — should I look something up, "
    "change a value, or make a report?")

_PROMPT = """You are routing a medical-records assistant. Classify the user's
latest request into exactly one intent, given the recent conversation.

Intents:
- generate_pdf: MAKE/CREATE/GENERATE a PDF/report OUT OF stored records.
- edit: CHANGE/CORRECT/FIX/UPDATE/SET an extracted value, name, or date.
- structured_query: ask for a specific document/record by patient, type, or recency.
- rag_query: a question about the CONTENT of documents.
- ingest: read/store a newly provided document.

Return JSON: {{"intent": <one of the above>, "confidence": <0..1>, "question": <a
short clarifying question if and only if you are unsure, else null>}}.
Set confidence below 0.8 only when the request is genuinely ambiguous.

Recent conversation:
{conversation}

JSON:"""


def _recent_conversation(state: dict[str, Any], n: int = 6) -> str:
    msgs = state.get("messages", [])[-n:]
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)


def _say(state: dict[str, Any], msg: str, **extra: Any) -> dict[str, Any]:
    return {"answer": msg,
            "messages": state["messages"] + [{"role": "assistant", "content": msg}],
            **extra}


def classify_intent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    # A pending file upload always means ingest.
    if state.get("file_path"):
        return {"intent": "ingest", "route_gate": "go"}

    deps = config["configurable"]["deps"]
    decision: IntentDecision = deps.chat.structured(
        _PROMPT.format(conversation=_recent_conversation(state)), IntentDecision)

    if decision.confidence < CLARIFY_BELOW:
        question = decision.question or _FALLBACK_QUESTION
        return _say(state, question, intent="clarify", route_gate="clarify")

    gate = "confirm" if decision.confidence < CONFIRM_BELOW else "go"
    return {"intent": decision.intent, "route_gate": gate}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/agent/test_router.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/router.py tests/agent/test_router.py
git commit -m "feat(router): confidence-scored routing with clarify/confirm/go gates"
```

---

### Task 3: `confirm_intent_node` HITL gate

**Files:**
- Modify: `app/agent/router.py` (add `confirm_intent_node` + `INTENT_ENTRY` map)
- Test: `tests/agent/test_router.py` (add node tests using a fake `interrupt`)

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_router.py`:

```python
def test_confirm_intent_approve(monkeypatch):
    import app.agent.router as router
    monkeypatch.setattr(router, "interrupt", lambda payload: {"approve": True})
    state = {"messages": [{"role": "user", "content": "show jane"}],
             "intent": "structured_query"}
    out = router.confirm_intent_node(state, _cfg(_FakeChat(IntentDecision(intent="rag_query"))))
    assert out["route_gate"] == "go"
    assert out["intent"] == "structured_query"


def test_confirm_intent_reject(monkeypatch):
    import app.agent.router as router
    monkeypatch.setattr(router, "interrupt", lambda payload: {"approve": False})
    state = {"messages": [{"role": "user", "content": "show jane"}],
             "intent": "structured_query"}
    out = router.confirm_intent_node(state, _cfg(_FakeChat(IntentDecision(intent="rag_query"))))
    assert out["route_gate"] == "clarify"
    assert out["messages"][-1]["role"] == "assistant"


def test_intent_entry_covers_all_intents():
    from app.agent.router import INTENT_ENTRY
    assert set(INTENT_ENTRY) == {
        "ingest", "structured_query", "rag_query", "edit", "generate_pdf"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_router.py -k "confirm_intent or intent_entry" -v`
Expected: FAIL — `AttributeError: module has no attribute 'confirm_intent_node'` / `INTENT_ENTRY`.

- [ ] **Step 3: Add the node and map**

In `app/agent/router.py`, add the import at top:

```python
from langgraph.types import interrupt
```

Add at module level (after the constants):

```python
INTENT_ENTRY = {
    "ingest": "dedup_check",
    "structured_query": "parse_filters",
    "rag_query": "require_patient",
    "edit": "plan_edit",
    "generate_pdf": "plan_report",
}

_INTENT_SUMMARY = {
    "ingest": "read and store the attached document",
    "structured_query": "look up a specific record",
    "rag_query": "answer a question from the document contents",
    "edit": "change an extracted value",
    "generate_pdf": "build a PDF report",
}
```

Add the node function at the end of the file:

```python
def confirm_intent_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Gate for medium-confidence (0.80-0.90) routing. Approve -> run the chain,
    reject -> ask the user to rephrase."""
    intent = state.get("intent") or "rag_query"
    decision = interrupt({
        "type": "confirm_intent",
        "intent": intent,
        "summary": _INTENT_SUMMARY.get(intent, intent),
    })
    if decision.get("approve"):
        return {"route_gate": "go", "intent": intent}
    return _say(state, "No problem — could you rephrase what you'd like me to do?",
                intent="clarify", route_gate="clarify")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_router.py -k "confirm_intent or intent_entry" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/agent/router.py tests/agent/test_router.py
git commit -m "feat(router): confirm_intent HITL gate + INTENT_ENTRY map"
```

---

### Task 4: Wire the gates into the graph

**Files:**
- Modify: `app/agent/graph.py:23-24` (`_route`), `:40` (register node), `:69-76` (router edges)
- Test: `tests/agent/test_graph.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_graph.py` (match the file's existing graph-construction style; the key assertions are routing behavior):

```python
def test_graph_clarify_ends_turn():
    from app.agent.graph import build_graph
    from app.agent.state import Deps, IntentDecision

    class _Chat:
        def complete(self, p): return ""
        def structured(self, p, s):
            return IntentDecision(intent="rag_query", confidence=0.2,
                                  question="What would you like?")

    deps = Deps(chat=_Chat(), vision=None, embedder=None, session_factory=None)
    g = build_graph()
    state = {"messages": [{"role": "user", "content": "do the thing"}]}
    out = g.invoke(state, {"configurable": {"deps": deps, "thread_id": "t1"}})
    assert out["messages"][-1]["content"] == "What would you like?"
    assert out.get("answer") == "What would you like?"


def test_graph_mid_confidence_interrupts_confirm_intent():
    from app.agent.graph import build_graph
    from app.agent.state import Deps, IntentDecision

    class _Chat:
        def complete(self, p): return ""
        def structured(self, p, s):
            return IntentDecision(intent="structured_query", confidence=0.85)

    deps = Deps(chat=_Chat(), vision=None, embedder=None, session_factory=None)
    g = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": "t2"}}
    g.invoke({"messages": [{"role": "user", "content": "show jane"}]}, cfg)
    snap = g.get_state(cfg)
    assert snap.next  # interrupted, not finished
    interrupts = [i for t in snap.tasks for i in t.interrupts]
    assert interrupts and interrupts[0].value["type"] == "confirm_intent"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/agent/test_graph.py -k "clarify or confirm_intent" -v`
Expected: FAIL — no `confirm_intent` node registered; clarify routes nowhere / runs a chain.

- [ ] **Step 3: Wire the graph**

In `app/agent/graph.py`:

Update the import line for the router:

```python
from app.agent.router import classify_intent, confirm_intent_node, INTENT_ENTRY
```

Replace `_route` (lines 23-24) with a gate-aware router:

```python
def _route(state: AgentState) -> str:
    gate = state.get("route_gate") or "go"
    if gate == "clarify":
        return "clarify"
    if gate == "confirm":
        return "confirm"
    return state.get("intent") or "rag_query"
```

Register the new node (after `g.add_node("router", classify_intent)`):

```python
    g.add_node("confirm_intent", confirm_intent_node)
```

Replace the router conditional edges (current lines 70-76) with:

```python
    g.add_conditional_edges("router", _route, {
        "clarify": END,
        "confirm": "confirm_intent",
        **INTENT_ENTRY,
    })
    g.add_conditional_edges("confirm_intent",
                            lambda s: "clarify" if s.get("route_gate") == "clarify"
                            else (s.get("intent") or "rag_query"),
                            {"clarify": END, **INTENT_ENTRY})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/agent/test_graph.py -k "clarify or confirm_intent" -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Run the full agent suite for regressions**

Run: `pytest tests/agent/ -v`
Expected: PASS (no regressions in existing graph/router/node tests)

- [ ] **Step 6: Commit**

```bash
git add app/agent/graph.py tests/agent/test_graph.py
git commit -m "feat(router): wire clarify + confirm_intent gates into graph"
```

---

### Task 5: Frontend `confirm_intent` interrupt card

**Files:**
- Modify: `medagentic-dashboard/src/main.ts` (interrupt renderer, near the `confirm_report` card ~line 783; resume button handler ~line 906)

- [ ] **Step 1: Add the card branch**

In `medagentic-dashboard/src/main.ts`, in the interrupt-rendering function (where `payload.type === 'confirm_report'` etc. are handled), add before the `low_confidence` fallback:

```javascript
  if (payload.type === 'confirm_intent') {
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#5D7B6F]">
          <i data-lucide="help-circle" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Just checking</span>
        </div>
        <p class="text-[13px] text-[#2E2C29] mb-5">I think you want me to ${esc(payload.summary)}. Go ahead?</p>
        <div class="flex gap-2.5">
          <button data-act="reject-intent" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">No, let me rephrase</button>
          <button data-act="approve-intent" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Yes, go ahead</button>
        </div>
      </div>`;
  }
```

- [ ] **Step 2: Handle the resume actions**

In the interrupt-button click handler (where `data-act` values like `confirm`/`cancel`/`regenerate` are dispatched to `/api/chat/resume`), add cases mapping the new actions to the resume payload `{approve: bool}`:

```javascript
      } else if (act === 'approve-intent') {
        await resumeChat(threadId, { approve: true }, handlers);
      } else if (act === 'reject-intent') {
        await resumeChat(threadId, { approve: false }, handlers);
```

(Match the exact `resumeChat` call signature already used by the `confirm_report` approve/cancel buttons in this handler.)

- [ ] **Step 3: Build to verify it compiles**

Run: `cd medagentic-dashboard && npm run build`
Expected: build succeeds, no TS errors.

- [ ] **Step 4: Commit**

```bash
git add medagentic-dashboard/src/main.ts
git commit -m "feat(ui): confirm_intent interrupt card (approve/rephrase)"
```

---

### Task 6: Manual end-to-end verification

**Files:** none (manual)

- [ ] **Step 1: Start backend + frontend**

Run backend (FastAPI) and `cd medagentic-dashboard && npm run dev`. Open the app.

- [ ] **Step 2: Low-confidence clarify**

Pick a patient, type an ambiguous message ("do the thing"). Expect a plain assistant bubble asking a clarifying question, no chain run.

- [ ] **Step 3: Multi-turn fusion**

Reply with the clarification ("make a pdf of all reports"). Expect it to route to report generation, proving the router used recent history.

- [ ] **Step 4: Mid-confidence confirm**

Type a borderline request ("show jane") with an ambiguous patient name. Expect a `confirm_intent` card; Approve runs the chain, "No, let me rephrase" returns a clarify prompt.

- [ ] **Step 5: High-confidence passthrough**

Type a clear request ("latest report of <patient>"). Expect it to run directly with no extra gate.

---

## Self-Review Notes

- **Spec coverage:** confidence schema (T1), scored router + thresholds + multi-turn history (T2), confirm gate (T3), graph wiring incl. clarify→END (T4), UI card + plain-text clarify reuse (T5), tests throughout, manual E2E (T6). All spec sections mapped.
- **Type consistency:** `IntentDecision{intent,confidence,question}`, `route_gate` values `go|confirm|clarify`, `INTENT_ENTRY` keys match the five intents, interrupt payload `{type:"confirm_intent",intent,summary}` / resume `{approve:bool}` — consistent across tasks.
- **Known v1 limit (from spec):** chains still parse from the last user message; their own confirm gates mitigate. Not addressed here by design.
