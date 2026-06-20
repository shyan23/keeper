from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel
from sqlalchemy import func

from app.agent.state import Deps  # noqa: F401  (documents the dep contract)
from app.agent.nodes.ingest import _normalize_name
from app.models import Document, Patient


class _Filters(BaseModel):
    patient_name: str | None = None
    doc_type: str | None = None
    latest: bool = False


# Generic words that carry no report identity — dropped before matching so
# "lipid profile report" still hits a doc named just "Lipid Profile".
_NOISE = {"report", "reports", "test", "tests", "result", "results", "the", "of"}
_FUZZ = 0.8  # min SequenceMatcher ratio to call two words the same


_SUGGEST = 0.6  # looser floor for "did you mean …" when nothing clears _FUZZ


def _query_words(term: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", term.lower()) if w not in _NOISE]


def _hay_words(doc: Document) -> list[str]:
    hay = " ".join(filter(None, [doc.classification, doc.doc_type, doc.original_name]))
    return re.findall(r"[a-z0-9]+", hay.lower())


def _word_score(w: str, hw: str) -> float:
    """1.0 for equal or a ≥4-char containment (so 'cardio'⊂'cardiology'); else
    the SequenceMatcher ratio. The length guard stops a 1-char token like the
    'x' in 'X-Ray' from matching any word that happens to contain an 'x'."""
    if w == hw:
        return 1.0
    if len(w) >= 4 and len(hw) >= 4 and (w in hw or hw in w):
        return 1.0
    return SequenceMatcher(None, w, hw).ratio()


def _doc_score(term: str, doc: Document) -> float:
    """Mean over query words of the best per-word score against any doc word
    (category/type/filename). Tolerates spelling drift like
    haemotology/haematology/hematology that exact SQL LIKE misses."""
    words = _query_words(term)
    if not words:
        return 1.0
    hw = _hay_words(doc)
    if not hw:
        return 0.0
    return sum(max(_word_score(w, h) for h in hw) for w in words) / len(words)


def _doc_matches(term: str, doc: Document) -> bool:
    """A confident match: every query word clears the fuzzy bar."""
    words = _query_words(term)
    if not words:
        return True
    hw = _hay_words(doc)
    return all(any(_word_score(w, h) >= _FUZZ for h in hw) for w in words)


_PROMPT = """From the user's request extract: patient_name, doc_type (or null), and
latest (true if they want the most recent/last one). Request: {text}"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def parse_filters_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = deps.chat.structured(_PROMPT.format(text=_last_user_text(state)), _Filters, config=config)
    return {"query_filters": f.model_dump()}


def query_db_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = state["query_filters"]
    with deps.session_factory() as s:
        q = s.query(Document).join(Patient, Patient.id == Document.patient_id)
        # Patient scope: a named patient (honorific/title tolerant) wins; else fall
        # back to whichever patient is selected in the UI so "show me the latest
        # document" means "…for this patient".
        if f.get("patient_name"):
            want = _normalize_name(f["patient_name"])
            ids = [p.id for p in s.query(Patient).all() if _normalize_name(p.name) == want]
            q = q.filter(Document.patient_id.in_(ids or [-1]))
        elif state.get("patient_id"):
            q = q.filter(Document.patient_id == state["patient_id"])
        q = q.order_by(
            func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(),
            Document.id.desc(),
        )
        # Patient scope keeps this to a handful of rows, so fuzzy-match the report
        # name in Python (SQL LIKE can't bridge spelling variants) over the already
        # date-ordered list, then take the limit.
        # ponytail: O(docs_per_patient) scan; fine at this scale, swap to pg_trgm if a
        # patient ever has thousands of docs.
        term = f.get("doc_type")
        all_docs = q.all()  # already date-ordered, patient-scoped (a handful of rows)
        docs = [d for d in all_docs if not term or _doc_matches(term, d)]
        # No confident match? Don't dead-end on "No matching": offer the closest
        # doc above a looser floor as a "did you mean …" suggestion.
        suggested = False
        if term and not docs and all_docs:
            best = max(all_docs, key=lambda d: _doc_score(term, d))
            if _doc_score(term, best) >= _SUGGEST:
                docs, suggested = [best], True
        docs = docs[: 1 if f.get("latest") else 10]
        rows = [{"document_id": d.id, "doc_type": d.doc_type,
                 "name": d.original_name or f"document-{d.id}",
                 "date": (d.report_date.strftime("%Y-%m-%d") if d.report_date
                          else d.uploaded_at.strftime("%Y-%m-%d") if d.uploaded_at else None)}
                for d in docs]
    if not rows:
        msg = "No matching documents found."
        return {"answer": msg, "citations": [], "sources": [],
                "messages": state["messages"] + [{"role": "assistant", "content": msg, "sources": []}]}
    # User-facing text references documents by type + date — never internal ids.
    # The clickable document chips come from `sources` (rendered by the UI).
    if suggested:
        label = f"No exact match for “{term}” — did you mean"
    else:
        label = "Latest document" if f.get("latest") else f"Found {len(rows)} document" + ("s" if len(rows) != 1 else "")
    lines = [f"- {r['name']} · {r['doc_type'] or 'document'}{(' · ' + r['date']) if r['date'] else ''}"
             for r in rows]
    body = f"{label}:\n" + "\n".join(lines)
    return {"answer": body, "citations": rows, "sources": rows,
            "messages": state["messages"] + [{"role": "assistant", "content": body, "sources": rows}]}
