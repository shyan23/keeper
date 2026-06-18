import datetime as dt

from app.agent.state import Deps
from app.services.report import PdfRequest   # PdfRequest lives in the service, not state
from app.agent.nodes import report as rnode


class _FakeChat:
    def __init__(self, req: PdfRequest):
        self._req = req
    def complete(self, prompt):
        return ""
    def structured(self, prompt, schema):
        return self._req


def _cfg(sf, chat=None):
    deps = Deps(chat=chat, vision=None, embedder=None, session_factory=sf)
    return {"configurable": {"deps": deps}}


def test_plan_report_no_patient_dead_ends():
    chat = _FakeChat(PdfRequest(patient_name=None))
    out = rnode.plan_report_node({"messages": [{"role": "user", "content": "make a pdf"}]},
                                 _cfg(sf=None, chat=chat))
    assert out["report_plan"] is None
    assert "patient" in out["messages"][-1]["content"].lower()


def test_plan_report_builds_plan(db_session_factory):
    from app.services.patients import create_patient
    from app.models import Document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Plan Patient")
        s.add(Document(patient_id=p.id, doc_type="lipid profile",
                       report_date=dt.date(2022, 5, 1), original_name="lipid.pdf"))
        s.commit(); pid = p.id
    chat = _FakeChat(PdfRequest(patient_name="Plan Patient", doc_types=["lipid profile"]))
    state = {"messages": [{"role": "user", "content": "pdf of lipid profile"}]}
    out = rnode.plan_report_node(state, _cfg(sf=sf, chat=chat))
    plan = out["report_plan"]
    assert plan is not None
    assert plan["patient_id"] == pid
    assert len(plan["documents"]) == 1


def test_confirm_report_reject_cancels(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [{"role": "user", "content": "x"}],
             "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {}}
    rmod.interrupt = lambda payload: {"approved": False}
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "end"
    assert "cancel" in out["messages"][-1]["content"].lower()


def test_confirm_report_modify_replans(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [{"role": "user", "content": "x"}],
             "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {"last_n_years": 3}}
    rmod.interrupt = lambda payload: {"approved": True, "modify": {"last_n_years": 5}}
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "replan"
    assert out["report_request"]["last_n_years"] == 5


def test_confirm_report_approve_builds(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [], "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {}}
    rmod.interrupt = lambda payload: {"approved": True}
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "build"
