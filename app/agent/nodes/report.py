"""PDF report generation nodes. Two HITL gates: confirm_report (plan) and
deliver_report (delivery). Between them, build_report runs automatically."""
from __future__ import annotations

import datetime as dt
from pathlib import Path  # noqa: F401
from typing import Any

from langgraph.types import interrupt

import app.storage as storage  # noqa: F401
from app.agent.nodes.ingest import _normalize_name
from app.models import Patient
from app.services import charts as charts_svc  # noqa: F401
from app.services import pdf as pdf_svc  # noqa: F401
from app.services import report as report_svc
from app.services import trends  # noqa: F401

_SECTIONS = ["Cover", "Patient Information", "Disease Summary", "Symptoms Summary",
             "Medical Test Results", "Charts & Trends", "Timeline", "Attachments"]


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _say(state: dict[str, Any], msg: str, **extra: Any) -> dict[str, Any]:
    return {"answer": msg,
            "messages": state["messages"] + [{"role": "assistant", "content": msg}],
            **extra}


def _resolve_patient(s, name: str | None, fallback_id: int | None) -> int | None:
    if name:
        want = _normalize_name(name)
        for p in s.query(Patient).all():
            if _normalize_name(p.name) == want:
                return p.id
        return None
    return fallback_id


def _timeframe_label(lo, hi) -> str:
    if lo is None and hi is None:
        return "All records"
    return f"{lo.isoformat() if lo else '...'} - {hi.isoformat() if hi else '...'}"


def plan_report_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    req = state.get("report_request")
    if req is None:
        req = report_svc.parse_request(deps.chat, _last_user_text(state)).model_dump()
    today = dt.date.today()
    lo, hi = report_svc.resolve_timeframe(req, today)
    # Fast-fail: no name and no pre-selected patient — no DB query needed.
    if not req.get("patient_name") and not state.get("patient_id"):
        return _say(state, "Which patient is this report for? I couldn't match a "
                           "name and none is selected.",
                    report_plan=None, report_decision=None)
    with deps.session_factory() as s:
        pid = _resolve_patient(s, req.get("patient_name"), state.get("patient_id"))
        if not pid:
            return _say(state, "Which patient is this report for? I couldn't match a "
                               "name and none is selected.",
                        report_plan=None, report_decision=None)
        data = report_svc.gather(s, pid, req.get("doc_types") or [], lo, hi)
    if not data["documents"]:
        return _say(state, "No documents match that patient and timeframe.",
                    report_plan=None, report_decision=None)
    plan = {
        "patient_id": pid,
        "patient_name": data["patient_name"],
        "timeframe_label": _timeframe_label(lo, hi),
        "date_from": lo.isoformat() if lo else None,
        "date_to": hi.isoformat() if hi else None,
        "doc_types": req.get("doc_types") or [],
        "documents": [{"name": d.get("original_name") or f"document-{d['id']}",
                       "type": d.get("type"),
                       "date": d.get("report_date") or d.get("date")}
                      for d in data["documents"]],
        "counts": {"documents": len(data["documents"]),
                   "diseases": len(data["diseases"]),
                   "tests": len(data["tests"]),
                   "attachments": len(data["attachments"])},
    }
    return {"report_request": req, "report_plan": plan, "report_decision": None}


def confirm_report_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Gate A. Show the interpreted patient/timeframe/documents and wait. Approve
    -> build; Modify -> re-plan with edits; Reject -> end."""
    plan = state.get("report_plan")
    if not plan:
        return {}
    decision = interrupt({"type": "confirm_report", "plan": plan})
    if not decision.get("approved"):
        return _say(state, "Report cancelled - nothing generated.", report_decision="end")
    mods = decision.get("modify")
    if mods:
        return {"report_decision": "replan",
                "report_request": {**(state.get("report_request") or {}), **mods},
                "report_plan": None}
    return {"report_decision": "build"}
