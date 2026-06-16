from app.agent.state import Deps
from app.agent.nodes.rag import (
    grade_node, generate_answer_node,
    transform_query_node, rerank_node, correct_query_node, confirm_low_confidence_node,
)


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


class _ScoreByContentChat:
    """complete() returns a high score when the snippet looks relevant, else low."""
    def complete(self, prompt):
        return "0.9" if "good" in prompt else "0.1"

    def structured(self, prompt, schema):
        raise NotImplementedError


def test_transform_query_sets_retrieval_query():
    state = {"messages": [{"role": "user", "content": "what is her hemoglobin?"}]}
    out = transform_query_node(state, _cfg(_FakeChat("hypothetical hemoglobin doc")))
    assert out["retrieval_query"] == "hypothetical hemoglobin doc"


def test_rerank_orders_by_score_and_trims():
    state = {
        "messages": [{"role": "user", "content": "q"}],
        "retrieved": [
            {"chunk_id": 1, "text": "bad snippet"},
            {"chunk_id": 2, "text": "good snippet"},
        ],
    }
    out = rerank_node(state, _cfg(_ScoreByContentChat()))
    assert out["retrieved"][0]["chunk_id"] == 2  # 'good' ranked first


def test_correct_query_sets_corrected_flag():
    state = {"messages": [{"role": "user", "content": "sugar?"}]}
    out = correct_query_node(state, _cfg(_FakeChat("blood glucose level value")))
    assert out["corrected"] is True
    assert out["retrieval_query"] == "blood glucose level value"


def test_confirm_low_conf_skips_when_empty_hits():
    out = confirm_low_confidence_node({"low_confidence": True, "retrieved": []}, _cfg(_FakeChat("x")))
    assert out == {}
