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
