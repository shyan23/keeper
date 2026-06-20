from app.agent.state import Deps
from app.agent.nodes.rag import (
    grade_node, generate_answer_node,
    transform_query_node, rerank_node, correct_query_node, confirm_low_confidence_node,
)


class _FakeChat:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt, config=None):
        return self._text

    def structured(self, prompt, schema, config=None):
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


def test_generate_answer_emits_doc_sources_not_chunk_ids():
    state = {
        "messages": [{"role": "user", "content": "eosinophil?"}],
        "retrieved": [
            {"chunk_id": 50, "document_id": 7, "text": "Eosinophil 8%", "doc_type": "LAB REPORT",
             "report_date": "2021-04-30", "original_name": "lab.pdf", "page_ref": "1"},
            {"chunk_id": 46, "document_id": 7, "text": "more", "doc_type": "LAB REPORT",
             "report_date": "2021-04-30", "original_name": "lab.pdf", "page_ref": "2"},
        ],
    }
    out = generate_answer_node(state, _cfg(_FakeChat("Eosinophil is 8%.")))
    last = out["messages"][-1]
    # no chunk ids, no glued raw-OCR "Sources:" block
    assert "#50" not in last["content"] and "#46" not in last["content"]
    assert "Sources:" not in last["content"]
    # chunks of one document collapse into a single citation
    assert len(out["sources"]) == 1
    assert out["sources"][0]["document_id"] == 7
    assert out["sources"][0]["date"] == "2021-04-30"
    assert last["sources"] == out["sources"]


def test_generate_answer_refuses_when_empty():
    state = {"messages": [{"role": "user", "content": "x"}], "retrieved": []}
    out = generate_answer_node(state, _cfg(_FakeChat("ignored")))
    assert "don't have" in out["answer"].lower() or "no relevant" in out["answer"].lower()


class _ScoreByContentChat:
    """complete() returns a high score when the snippet looks relevant, else low."""
    def complete(self, prompt, config=None):
        return "0.9" if "good" in prompt else "0.1"

    def structured(self, prompt, schema, config=None):
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


def test_require_patient_passes_when_set():
    from app.agent.nodes.rag import require_patient_node
    out = require_patient_node({"patient_id": 3}, _cfg(_FakeChat("")))
    assert out == {}


def test_require_patient_asks_then_resumes(monkeypatch):
    """No patient -> interrupt with a picker; the resumed choice becomes patient_id."""
    import app.agent.nodes.rag as rag_mod
    from app.agent.nodes.rag import require_patient_node

    class _FakeSession:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def query(self, *a): return self
        def order_by(self, *a): return self
        def all(self): return []

    captured = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return {"patient_id": 9}  # simulate the human's pick on resume

    monkeypatch.setattr(rag_mod, "interrupt", fake_interrupt)
    cfg = _cfg(_FakeChat(""))
    cfg["configurable"]["deps"].session_factory = lambda: _FakeSession()
    out = require_patient_node({"patient_id": None}, cfg)
    assert captured["payload"]["type"] == "patient_pick"
    assert out == {"patient_id": 9}
