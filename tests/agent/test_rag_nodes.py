from app.agent.state import Deps
from app.agent.nodes.rag import grade_node, generate_answer_node


class _FakeChat:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt):
        return self._text

    def structured(self, prompt, schema):
        raise NotImplementedError


def _cfg(chat):
    return {"configurable": {"deps": Deps(chat=chat, vision=None, embedder=None, session_factory=None)}}


def test_grade_low_confidence_flags():
    state = {"retrieved": [{"chunk_id": 1, "text": "x"}]}
    out = grade_node(state, _cfg(_FakeChat("0.1")))
    assert out["low_confidence"] is True


def test_grade_high_confidence_passes():
    state = {"retrieved": [{"chunk_id": 1, "text": "x"}]}
    out = grade_node(state, _cfg(_FakeChat("0.92")))
    assert out["low_confidence"] is False


def test_generate_answer_includes_citations():
    state = {
        "messages": [{"role": "user", "content": "what is her hemoglobin?"}],
        "retrieved": [
            {"chunk_id": 7, "text": "hemoglobin 13.5 g/dL", "doc_type": "lab_report", "uploaded_at": "2026-06-10"},
        ],
    }
    out = generate_answer_node(state, _cfg(_FakeChat("Her hemoglobin is 13.5 g/dL")))
    assert out["answer"]
    assert out["citations"][0]["chunk_id"] == 7
    assert "#7" in out["messages"][-1]["content"]


def test_generate_answer_refuses_when_empty():
    state = {"messages": [{"role": "user", "content": "x"}], "retrieved": []}
    out = generate_answer_node(state, _cfg(_FakeChat("ignored")))
    assert "don't have" in out["answer"].lower() or "no relevant" in out["answer"].lower()
