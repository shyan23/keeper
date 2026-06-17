from app.api import runtime


def test_node_labels_cover_key_nodes():
    for node in ["router", "extract_text", "generate_answer"]:
        assert node in runtime.NODE_LABELS


def test_cfg_has_thread_and_progress():
    calls = []
    cfg = runtime.cfg("thread-abc", deps={"x": 1}, progress=calls.append)
    assert cfg["configurable"]["thread_id"] == "thread-abc"
    assert cfg["configurable"]["deps"] == {"x": 1}
    cfg["configurable"]["progress"]("hi")
    assert calls == ["hi"]


from langgraph.types import Command

from app.api import sse


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeGraph:
    """Scriptable stand-in for the compiled LangGraph."""

    def __init__(self, chunks, final_messages=None):
        self._chunks = chunks
        self._final = final_messages or []
        self.last_input = None

    def stream(self, payload, cfg, stream_mode="updates"):
        self.last_input = payload
        for ch in self._chunks:
            yield ch

    def get_state(self, cfg):
        class S:
            values = {"messages": self._final}
        return S()


def _collect(gen):
    return "".join(line for line in gen if line.strip())


def test_sse_clean_answer_sequence():
    graph = FakeGraph(
        chunks=[{"router": {}}, {"generate_answer": {}}],
        final_messages=[{"role": "assistant", "content": "Hi", "sources": ["a.pdf"]}],
    )
    body = _collect(sse.run_graph_sse(graph, {"messages": []}, "t1", deps={}))
    assert "event: node" in body
    assert "🧠 Composing the answer" in body
    assert "event: message" in body
    assert "Hi" in body
    assert "a.pdf" in body
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_sse_interrupt_then_resume():
    graph = FakeGraph(
        chunks=[{"extract_text": {}},
                {"__interrupt__": (_FakeInterrupt({"type": "confirm_ingest",
                                                   "extracted": {}}),)}],
    )
    body = _collect(sse.run_graph_sse(graph, {"messages": []}, "t2", deps={}))
    assert "event: interrupt" in body
    assert "confirm_ingest" in body
    assert "event: message" not in body  # paused: no final message

    graph2 = FakeGraph(chunks=[{"persist": {}}],
                       final_messages=[{"role": "assistant", "content": "done"}])
    body2 = _collect(sse.run_graph_sse(graph2, Command(resume={"approved": True}),
                                       "t2", deps={}))
    assert "💾 Saving entities" in body2
    assert "event: message" in body2
    assert isinstance(graph2.last_input, Command)


def test_sse_error_event():
    class Boom(FakeGraph):
        def stream(self, payload, cfg, stream_mode="updates"):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    body = _collect(sse.run_graph_sse(Boom([]), {"messages": []}, "t3", deps={}))
    assert "event: error" in body
    assert "provider down" in body
    assert "event: done" in body


import app.api.runtime as runtime_mod
from fastapi.testclient import TestClient


def test_upload_stages_file():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/upload",
                    files={"file": ("scan.png", b"\x89PNG fake", "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ext"] == "png"
    assert body["mime"] == "image/png"
    assert body["staged_path"].endswith(".png")


def test_upload_rejects_unsupported_ext():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/upload",
                    files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_stream_text_requires_patient():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/stream",
                    json={"thread_id": "t-np", "message": "hello"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "pick a patient" in r.text.lower()


def test_stream_runs_graph(monkeypatch):
    from app.api import server
    client = TestClient(server.app)

    class S:
        values = {"messages": [{"role": "assistant", "content": "Answer", "sources": []}]}

    class G:
        def stream(self, payload, cfg, stream_mode="updates"):
            yield {"router": {}}
            yield {"generate_answer": {}}
        def get_state(self, cfg):
            return S()

    monkeypatch.setattr(runtime_mod, "get_graph", lambda: G())
    monkeypatch.setattr(runtime_mod, "get_deps", lambda: {"fake": True})

    r = client.post("/api/chat/stream",
                    json={"thread_id": "t-ok", "message": "hi", "patient_id": 1})
    assert r.status_code == 200
    assert "event: message" in r.text
    assert "Answer" in r.text
    assert r.text.rstrip().endswith("event: done\ndata: {}")
