import uuid
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.state import Deps, ExtractionResult, ExtractedEntity, IntentDecision


class _FakeChat:
    def __init__(self, label):
        self._label = label

    def complete(self, prompt, config=None):
        return self._label

    def structured(self, prompt, schema, config=None):
        if schema is ExtractionResult:
            return ExtractionResult(
                patient_name="Graph Pt", doc_type="prescription",
                diseases=[ExtractedEntity(name="flu", confidence=0.9, source_span="flu")],
            )
        if schema is IntentDecision:
            valid = {"ingest", "structured_query", "rag_query", "edit", "generate_pdf"}
            intent = self._label if self._label in valid else "rag_query"
            return IntentDecision(intent=intent, confidence=0.95)
        return schema()


class _FakeVision:
    def ocr_image(self, data, mime):
        return "Patient Graph Pt, flu"


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_ingest_existing_patient_creates_document(db_session_factory, tmp_path):
    """Upload for an already-known patient: pauses at the single confirm_ingest
    gate, then the agent creates the document under the matched patient."""
    from app.services.patients import create_patient
    from app.models import Document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Graph Pt")
        pid = p.id
    f = tmp_path / "rx.png"
    f.write_bytes(b"\x89PNG")
    deps = Deps(chat=_FakeChat("ingest"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    # NOTE: no document_id / patient_id passed — the agent determines the patient.
    state = {"messages": [{"role": "user", "content": "read this"}],
             "file_path": str(f), "mime_type": "image/png", "file_ext": "png",
             "source_type": "image"}

    result = graph.invoke(state, cfg, stream_mode="updates")
    assert result[-1]["__interrupt__"][0].value["type"] == "confirm_ingest"

    final = graph.invoke(Command(resume={"approved": True}), cfg)
    assert final["patient_id"] == pid       # matched "Graph Pt" by name in one gate
    assert final.get("document_id")         # document created by the agent
    assert any("Indexed" in m["content"] for m in final["messages"])
    with sf() as s:
        assert s.get(Document, final["document_id"]).patient_id == pid


def test_ingest_new_patient_full_agentic_flow(db_session_factory, tmp_path):
    """Upload for an UNKNOWN patient: ONE gate reviews patient + entities together.
    The agent extracts the name; approval creates the profile and arranges the
    document + entities under it."""
    from app.models import Patient, Document
    sf = db_session_factory
    f = tmp_path / "rx.png"
    f.write_bytes(b"\x89PNG")
    deps = Deps(chat=_FakeChat("ingest"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    state = {"messages": [{"role": "user", "content": "read this and arrange it"}],
             "file_path": str(f), "mime_type": "image/png", "file_ext": "png",
             "source_type": "image"}

    r1 = graph.invoke(state, cfg, stream_mode="updates")
    assert r1[-1]["__interrupt__"][0].value["type"] == "confirm_ingest"  # single gate

    final = graph.invoke(Command(resume={"approved": True}), cfg)
    assert final.get("patient_id") and final.get("document_id")
    with sf() as s:
        pat = s.get(Patient, final["patient_id"])
        assert pat.name == "Graph Pt"  # profile built from the extracted name
        assert s.get(Document, final["document_id"]).patient_id == final["patient_id"]
    assert any("Indexed" in m["content"] for m in final["messages"])


def test_rag_query_runs_without_pause(db_session_factory):
    from app.services.patients import create_patient
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Rag Pt")
        pid = p.id
    deps = Deps(chat=_FakeChat("0.9"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    state = {"messages": [{"role": "user", "content": "what about flu?"}],
             "patient_id": pid}
    out = graph.invoke(state, cfg)
    assert out["answer"]


def test_graph_has_report_nodes():
    g = build_graph()
    nodes = set(g.get_graph().nodes)
    for n in ["plan_report", "confirm_report", "build_report", "deliver_report"]:
        assert n in nodes


def test_graph_clarify_ends_turn():
    from app.agent.graph import build_graph
    from app.agent.state import Deps, IntentDecision

    class _Chat:
        def complete(self, p, config=None): return ""
        def structured(self, p, s, config=None):
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
        def complete(self, p, config=None): return ""
        def structured(self, p, s, config=None):
            return IntentDecision(intent="structured_query", confidence=0.85)

    deps = Deps(chat=_Chat(), vision=None, embedder=None, session_factory=None)
    g = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": "t2"}}
    g.invoke({"messages": [{"role": "user", "content": "show jane"}]}, cfg)
    snap = g.get_state(cfg)
    assert snap.next  # interrupted, not finished
    interrupts = [i for t in snap.tasks for i in t.interrupts]
    assert interrupts and interrupts[0].value["type"] == "confirm_intent"
