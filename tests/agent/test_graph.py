import uuid
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.state import Deps, ExtractionResult, ExtractedEntity


class _FakeChat:
    def __init__(self, label):
        self._label = label

    def complete(self, prompt):
        return self._label

    def structured(self, prompt, schema):
        if schema is ExtractionResult:
            return ExtractionResult(
                patient_name="Graph Pt", doc_type="prescription",
                diseases=[ExtractedEntity(name="flu", confidence=0.9, source_span="flu")],
            )
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
    """Upload for an already-known patient: pauses at confirm_entities, then the
    agent (not the user) creates the document under the matched patient."""
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
    assert "__interrupt__" in result[-1]  # paused at confirm_entities

    final = graph.invoke(Command(resume={"approved": True}), cfg)
    assert final["patient_id"] == pid       # matched "Graph Pt" by name, no patient gate
    assert final.get("document_id")         # document created by the agent
    assert any("Indexed" in m["content"] for m in final["messages"])
    with sf() as s:
        assert s.get(Document, final["document_id"]).patient_id == pid


def test_ingest_new_patient_full_agentic_flow(db_session_factory, tmp_path):
    """Upload for an UNKNOWN patient: two gates (entities, then create-profile).
    The agent extracts the name, the human approves a new profile, then the
    document + entities are arranged under it."""
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
    assert r1[-1]["__interrupt__"][0].value["type"] == "confirm_entities"

    r2 = graph.invoke(Command(resume={"approved": True}), cfg, stream_mode="updates")
    assert r2[-1]["__interrupt__"][0].value["type"] == "confirm_patient"  # name is new

    final = graph.invoke(Command(resume={"create_new": True}), cfg)
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
