from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy import func

from app.agent.state import Deps  # noqa: F401  (documents the dep contract)
from app.models import Document, Patient


class _Filters(BaseModel):
    patient_name: str | None = None
    doc_type: str | None = None
    latest: bool = False


_PROMPT = """From the user's request extract: patient_name, doc_type (or null), and
latest (true if they want the most recent/last one). Request: {text}"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def parse_filters_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = deps.chat.structured(_PROMPT.format(text=_last_user_text(state)), _Filters)
    return {"query_filters": f.model_dump()}


def query_db_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = state["query_filters"]
    with deps.session_factory() as s:
        q = s.query(Document).join(Patient, Patient.id == Document.patient_id)
        if f.get("patient_name"):
            q = q.filter(func.lower(Patient.name) == f["patient_name"].lower())
        if f.get("doc_type"):
            q = q.filter(Document.doc_type == f["doc_type"])
        q = q.order_by(
            func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(),
            Document.id.desc(),
        )
        limit = 1 if f.get("latest") else 10
        docs = q.limit(limit).all()
        rows = [{"document_id": d.id, "doc_type": d.doc_type,
                 "name": d.original_name or f"document-{d.id}",
                 "date": (d.report_date.strftime("%Y-%m-%d") if d.report_date
                          else d.uploaded_at.strftime("%Y-%m-%d") if d.uploaded_at else None)}
                for d in docs]
    if not rows:
        return {"answer": "No matching documents found.", "citations": []}
    # User-facing text references documents by type + date — never internal ids.
    lines = [f"- {r['doc_type'] or 'document'}{(' — ' + r['date']) if r['date'] else ''}"
             for r in rows]
    return {"answer": "Found:\n" + "\n".join(lines), "citations": rows}
