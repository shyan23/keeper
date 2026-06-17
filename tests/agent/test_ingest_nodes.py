from app.agent.state import Deps, ExtractionResult, ExtractedEntity
from app.agent.nodes.ingest import (
    extract_text_node, extract_entities_node, resolve_patient_node,
)


class _FakeVision:
    def ocr_image(self, data, mime):
        return "Patient Jane Doe, Dx hypertension"


class _FakeChat:
    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
        return ExtractionResult(
            patient_name="Jane Doe", doc_type="prescription",
            diseases=[ExtractedEntity(name="hypertension", confidence=0.9, source_span="Dx hypertension")],
        )


def _cfg(**kw):
    deps = Deps(chat=kw.get("chat"), vision=kw.get("vision"),
                embedder=kw.get("embedder"), session_factory=kw.get("sf"))
    return {"configurable": {"deps": deps}}


def test_extract_text_node_reads_bytes(tmp_path):
    f = tmp_path / "scan.png"
    f.write_bytes(b"\x89PNG fake")
    state = {"file_path": str(f), "mime_type": "image/png"}
    out = extract_text_node(state, _cfg(vision=_FakeVision()))
    assert "Jane Doe" in out["ocr_text"]


def test_extract_entities_node_returns_dict():
    state = {"ocr_text": "Patient Jane Doe, Dx hypertension"}
    out = extract_entities_node(state, _cfg(chat=_FakeChat()))
    assert out["extracted"]["patient_name"] == "Jane Doe"
    assert out["extracted"]["diseases"][0]["name"] == "hypertension"


def test_resolve_patient_exact_match(db_session_factory):
    from app.services.patients import create_patient
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Unique Resolve Name")
        pid = p.id
    state = {"extracted": {"patient_name": "Unique Resolve Name"}}
    out = resolve_patient_node(state, _cfg(sf=sf))
    assert out["patient_id"] == pid
    assert out.get("patient_candidates") in (None, [])


def test_resolve_patient_fuzzy_candidate(db_session_factory):
    """A near-but-not-exact name surfaces as a candidate (ask 'same as X?'), never auto-resolved."""
    from app.services.patients import create_patient
    sf = db_session_factory
    with sf() as s:
        create_patient(s, name="Nafisa Kabir Fuzzy")
    state = {"extracted": {"patient_name": "Nafisa Kabier Fuzzy"}}  # 1-char diff
    out = resolve_patient_node(state, _cfg(sf=sf))
    assert out["patient_id"] is None
    assert "Nafisa Kabir Fuzzy" in [c["name"] for c in out["patient_candidates"]]


def test_resolve_patient_no_match_sets_candidates(db_session_factory):
    state = {"extracted": {"patient_name": "Nobody Named This 9999"}}
    out = resolve_patient_node(state, _cfg(sf=db_session_factory))
    assert out["patient_id"] is None
    assert out["patient_candidates"] == []
