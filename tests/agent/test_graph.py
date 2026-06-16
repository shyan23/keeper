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


def test_ingest_pauses_at_confirm_entities(db_session_factory, tmp_path):
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Graph Pt")
        pid = p.id
        doc = create_document(s, patient_id=pid, doc_type="prescription")
        doc_id = doc.id
    f = tmp_path / "rx.png"
    f.write_bytes(b"\x89PNG")
    deps = Deps(chat=_FakeChat("ingest"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    state = {"messages": [{"role": "user", "content": "read this"}],
             "file_path": str(f), "mime_type": "image/png", "document_id": doc_id}

    # stream_mode="updates" accumulates into a list; the last element is
    # {"__interrupt__": (Interrupt(...),)} when the graph pauses
    result = graph.invoke(state, cfg, stream_mode="updates")
    assert "__interrupt__" in result[-1]  # paused at confirm_entities

    final = graph.invoke(Command(resume={"approved": True}), cfg)
    assert final["patient_id"] == pid
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
