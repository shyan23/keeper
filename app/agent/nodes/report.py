"""PDF report generation nodes. Two HITL gates: confirm_report (plan) and
deliver_report (delivery). Between them, build_report runs automatically."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import fitz
from langgraph.types import interrupt

import app.storage as storage
from app.agent.nodes.ingest import _normalize_name
from app.models import Patient
from app.services import charts as charts_svc
from app.services import pdf as pdf_svc
from app.services import report as report_svc
from app.services import trends

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


def _window(plan: dict) -> tuple[dt.date | None, dt.date | None]:
    lo = dt.date.fromisoformat(plan["date_from"]) if plan.get("date_from") else None
    hi = dt.date.fromisoformat(plan["date_to"]) if plan.get("date_to") else None
    return lo, hi


def _in_window_point(date_str: str, lo, hi) -> bool:
    d = dt.date.fromisoformat(date_str)
    return (lo is None or d >= lo) and (hi is None or d <= hi)


def build_report_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate -> charts -> assemble PDF -> store. Runs between the two gates."""
    deps = config["configurable"]["deps"]
    progress = config["configurable"].get("progress")
    req = state.get("report_request") or {}
    plan = state["report_plan"]
    pid = plan["patient_id"]
    lo, hi = _window(plan)

    if progress:
        progress("Aggregating records…")
    with deps.session_factory() as s:
        data = report_svc.gather(s, pid, req.get("doc_types") or [], lo, hi)
        data["timeframe_label"] = plan.get("timeframe_label")
        if progress:
            progress("Rendering charts…")
        charts: list[tuple[str, bytes]] = []
        for m in trends.list_metrics(s, pid):
            series = trends.metric_series(s, pid, m["key"])
            series["points"] = [p for p in series["points"]
                                if _in_window_point(p["date"], lo, hi)]
            if len(series["points"]) >= 2:
                charts.append((f"{m['label']} over time",
                               charts_svc.render_metric_chart(series)))
    if progress:
        progress("Assembling PDF…")
    pdf_bytes = pdf_svc.build_report(data, charts, data["attachments"])
    path = storage.save_report(pdf_bytes)
    url = f"/api/chat/report/{Path(path).name}"
    return {"report_path": path, "report_url": url, "report_decision": None,
            "report_plan": {**plan, "chart_count": len(charts),
                            "page_count": fitz.open("pdf", pdf_bytes).page_count}}


def deliver_report_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Gate B. Present the built report; Download finishes, Regenerate loops back."""
    url = state.get("report_url")
    plan = state.get("report_plan") or {}
    summary = {
        "url": url,
        "sections": _SECTIONS,
        "page_count": plan.get("page_count"),
        "chart_count": plan.get("chart_count"),
        "attachment_count": (plan.get("counts") or {}).get("attachments"),
    }
    decision = interrupt({"type": "confirm_delivery", "summary": summary})
    if decision.get("regenerate"):
        return {"report_decision": "rebuild"}
    return _say(state, f"Your report is ready. [Download the PDF]({url})",
                report_url=url, report_decision="end")
