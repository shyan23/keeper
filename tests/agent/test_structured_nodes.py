from pydantic import BaseModel
from app.agent.state import Deps
from app.agent.nodes.structured import parse_filters_node, query_db_node


class _FakeChat:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
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
    state = {"query_filters": {"patient_name": "Latest Query Pt", "doc_type": None, "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert "prescription" in out["answer"] or "lab_report" in out["answer"]
    assert out["citations"]
