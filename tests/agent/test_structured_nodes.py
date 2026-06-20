from pydantic import BaseModel
from app.agent.state import Deps
from app.agent.nodes.structured import parse_filters_node, query_db_node


class _FakeChat:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt, config=None):
        return ""

    def structured(self, prompt, schema, config=None):
        return schema(**self._payload)


def _cfg(chat=None, sf=None):
    return {"configurable": {"deps": Deps(chat=chat, vision=None, embedder=None, session_factory=sf)}}


def test_parse_filters_extracts_name_and_recency():
    state = {"messages": [{"role": "user", "content": "latest report of Jane Doe"}]}
    chat = _FakeChat({"patient_name": "Jane Doe", "doc_type": None, "latest": True})
    out = parse_filters_node(state, _cfg(chat=chat))
    assert out["query_filters"]["patient_name"] == "Jane Doe"
    assert out["query_filters"]["latest"] is True


def test_query_db_returns_latest_document(db_session_factory):
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Latest Query Pt")
        create_document(s, patient_id=p.id, doc_type="lab_report")
        create_document(s, patient_id=p.id, doc_type="prescription")
    state = {"messages": [],
             "query_filters": {"patient_name": "Latest Query Pt", "doc_type": None, "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert "prescription" in out["answer"] or "lab_report" in out["answer"]
    assert out["citations"]


def test_query_db_matches_report_name_per_word_picks_latest(db_session_factory):
    # Real bug: "lipid profile report" full-string-matched only the older doc
    # literally named "Lipid Profile Report", skipping the newer "Lipid Profile".
    # Per-word match (noise word "report" dropped) must pick the latest of both.
    import datetime as dt
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Lipid Pt")
        create_document(s, patient_id=p.id, doc_type="lab",
                        original_name="Lipid Profile Report",
                        report_date=dt.date(2020, 10, 5))
        create_document(s, patient_id=p.id, doc_type="lab",
                        original_name="Lipid Profile",
                        report_date=dt.date(2021, 2, 25))
    state = {"messages": [],
             "query_filters": {"patient_name": "Lipid Pt",
                               "doc_type": "lipid profile report", "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert "2021-02-25" in out["answer"]  # newest, not the 2020 "…Report"


def test_query_db_fuzzy_matches_misspelled_report(db_session_factory):
    # Spelling drift must still match: "haemotology" -> "Haematology".
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Fuzz Pt")
        create_document(s, patient_id=p.id, doc_type="lab",
                        classification="Haematology", original_name="Haematology Report")
    state = {"messages": [],
             "query_filters": {"patient_name": "Fuzz Pt",
                               "doc_type": "haemotology report", "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert out["citations"]


def test_query_db_suggests_closest_when_no_confident_match(db_session_factory):
    # A near-miss (one solid report on file) yields a "did you mean" suggestion
    # instead of a dead-end "No matching documents found.".
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Sugg Pt")
        create_document(s, patient_id=p.id, doc_type="lab",
                        classification="Immunology", original_name="Immunology Report")
    state = {"messages": [],
             "query_filters": {"patient_name": "Sugg Pt",
                               "doc_type": "imnlgy", "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert out["citations"]
    assert "did you mean" in out["answer"].lower()
