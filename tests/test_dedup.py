import hashlib

from app.agent.nodes.ingest import dedup_check_node


class _FakeSession:
    def __init__(self, existing): self._existing = existing
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Deps:
    def __init__(self, existing): self._existing = existing
    def session_factory(self):
        return _FakeSession(self._existing)


def _write(tmp_path, data):
    f = tmp_path / "scan.png"
    f.write_bytes(data)
    return str(f)


def test_dedup_hit_short_circuits(tmp_path, monkeypatch):
    import app.agent.nodes.ingest as ing

    class Doc:  # stand-in for an existing Document row
        id = 11
        patient_id = 7

    monkeypatch.setattr(ing, "find_by_content_hash", lambda s, h: Doc())
    path = _write(tmp_path, b"PNGDATA")
    state = {"file_path": path, "messages": [{"role": "user", "content": "x"}]}
    out = dedup_check_node(state, {"configurable": {"deps": _Deps(Doc())}})
    assert out["already_ingested"] is True
    assert out["document_id"] == 11
    assert out["patient_id"] == 7
    assert out["dedup"] == "duplicate"


def test_dedup_miss_passes_hash_through(tmp_path, monkeypatch):
    import app.agent.nodes.ingest as ing
    monkeypatch.setattr(ing, "find_by_content_hash", lambda s, h: None)
    data = b"FRESHDATA"
    path = _write(tmp_path, data)
    out = dedup_check_node({"file_path": path, "messages": []},
                           {"configurable": {"deps": _Deps(None)}})
    assert out["dedup"] == "new"
    assert out["content_hash"] == hashlib.sha256(data).hexdigest()
