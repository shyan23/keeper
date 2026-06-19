import datetime as dt

import fitz

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


def test_build_report_writes_pdf(db_session_factory, monkeypatch):
    from app.services.patients import create_patient
    from app.models import Document
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Build Patient", age=60)
        s.add(Document(patient_id=p.id, doc_type="lab report",
                       report_date=dt.date(2023, 1, 1), original_name="lab.pdf"))
        s.commit(); pid = p.id
    state = {"messages": [],
             "report_request": {"doc_types": []},
             "report_plan": {"patient_id": pid, "date_from": None, "date_to": None,
                             "timeframe_label": "All records"}}
    out = rmod.build_report_node(state, _cfg(sf=sf))
    assert out["report_url"].startswith("/api/chat/report/")
    assert out["report_path"].endswith(".pdf")
    assert fitz.open(out["report_path"]).page_count >= 1


def test_build_report_chart_mandatory_outside_window(db_session_factory):
    """A trend graph is mandatory: even when every point predates the requested
    window, charts fall back to the metric's full history (>=2 points)."""
    from app.services.patients import create_patient
    from app.models import Document, DocumentEntity, MedicalTest, TestResult
    import app.agent.nodes.report as rmod
    sf = db_session_factory

    def _add(s, pid, d, val):
        doc = Document(patient_id=pid, doc_type="lab report", report_date=d,
                       original_name="cbc.pdf")
        s.add(doc); s.flush()
        mt = MedicalTest(name="Haemoglobin"); s.add(mt); s.flush()
        tr = TestResult(medical_test_id=mt.id, value=val, unit="g/dL",
                        reference_range="12-16")
        s.add(tr); s.flush()
        s.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                             entity_id=tr.id))

    with sf() as s:
        p = create_patient(s, name="Chart Patient", age=50)
        _add(s, p.id, dt.date(2020, 10, 5), "13")
        _add(s, p.id, dt.date(2021, 4, 30), "11")
        s.commit(); pid = p.id
    # Window is "last 3 years" of 2026 -> excludes the 2020/2021 points entirely.
    state = {"messages": [],
             "report_request": {"doc_types": []},
             "report_plan": {"patient_id": pid,
                             "date_from": "2023-06-19", "date_to": "2026-06-19",
                             "timeframe_label": "2023-06-19 - 2026-06-19"}}
    out = rmod.build_report_node(state, _cfg(sf=sf))
    assert out["report_plan"]["chart_count"] >= 1


def test_deliver_report_download_finishes(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [], "report_url": "/api/chat/report/abc.pdf",
             "report_plan": {"counts": {"attachments": 2}, "chart_count": 1}}
    rmod.interrupt = lambda payload: {"approved": True}        # type: ignore
    out = rmod.deliver_report_node(state, _cfg(sf=sf))
    assert "/api/chat/report/abc.pdf" in out["messages"][-1]["content"]
    assert out.get("report_decision") in (None, "end")


def test_deliver_report_regenerate_loops(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [], "report_url": "/api/chat/report/abc.pdf",
             "report_plan": {"counts": {}, "chart_count": 0}}
    rmod.interrupt = lambda payload: {"regenerate": True}      # type: ignore
    out = rmod.deliver_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "rebuild"
