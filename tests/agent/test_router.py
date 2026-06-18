from app.agent.router import classify_intent


class _FakeChat:
    def __init__(self, label):
        self._label = label

    def complete(self, prompt):
        return self._label

    def structured(self, prompt, schema):
        raise NotImplementedError


def _cfg(chat):
    from app.agent.state import Deps
    deps = Deps(chat=chat, vision=None, embedder=None, session_factory=None)
    return {"configurable": {"deps": deps}}


def test_router_ingest_when_file_present():
    state = {"messages": [{"role": "user", "content": "read this"}], "file_path": "/x.png"}
    out = classify_intent(state, _cfg(_FakeChat("rag_query")))
    assert out["intent"] == "ingest"  # file present forces ingest


def test_router_uses_llm_label_for_text():
    state = {"messages": [{"role": "user", "content": "latest report of Jane"}]}
    out = classify_intent(state, _cfg(_FakeChat("structured_query")))
    assert out["intent"] == "structured_query"


def test_router_defaults_to_rag_on_garbage():
    state = {"messages": [{"role": "user", "content": "what about her sugar?"}]}
    out = classify_intent(state, _cfg(_FakeChat("nonsense")))
    assert out["intent"] == "rag_query"


def test_router_generate_pdf():
    state = {"messages": [{"role": "user", "content": "make a pdf of Jane's lipid results"}]}
    out = classify_intent(state, _cfg(_FakeChat("generate_pdf")))
    assert out["intent"] == "generate_pdf"
