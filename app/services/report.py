"""Aggregate already-extracted DB data into a report payload, plus parse the
NL PDF request and resolve its timeframe. Pure functions over a DB session — no
HTTP, no rendering, no LLM (except parse_request, which takes an injected chat
client). This is the single data source of truth for the PDF pipeline."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.agent.nodes.structured import _FUZZ, _query_words, _word_score
from app.models import Patient
from app.services import browse as bsvc
from app.services.dates import parse_doc_date


class PdfRequest(BaseModel):
    patient_name: str | None = None
    doc_types: list[str] = Field(default_factory=list)   # [] = all types
    years: list[int] = Field(default_factory=list)        # explicit years
    last_n_years: int | None = None
    last_n_months: int | None = None


_PARSE_PROMPT = """The user wants to generate a PDF report from medical records.
Extract:
- patient_name: the patient named, or null to use the currently selected patient
- doc_types: list of document/test types requested (e.g. ["lipid profile"], ["prescription"]); [] if "all reports"
- years: explicit calendar years mentioned (e.g. [2021, 2022]); [] if none
- last_n_years: integer if they said "last N years"; null otherwise
- last_n_months: integer if they said "last N months"; null otherwise
Request: {text}"""


def parse_request(chat: Any, text: str) -> PdfRequest:
    return chat.structured(_PARSE_PROMPT.format(text=text), PdfRequest)


def _shift_years(d: dt.date, n: int) -> dt.date:
    try:
        return d.replace(year=d.year - n)
    except ValueError:  # Feb 29 in a non-leap target year
        return d.replace(year=d.year - n, day=28)


def _shift_months(d: dt.date, n: int) -> dt.date:
    total = (d.year * 12 + (d.month - 1)) - n
    year, month = divmod(total, 12)
    month += 1
    # clamp day to the target month's length (avoid e.g. Mar 31 -> Feb)
    for day in range(d.day, 0, -1):
        try:
            return dt.date(year, month, day)
        except ValueError:
            continue
    return dt.date(year, month, 1)


def resolve_timeframe(req: dict, today: dt.date) -> tuple[dt.date | None, dt.date | None]:
    """(date_from, date_to) inclusive, or (None, None) for all-time. Years win,
    then last_n_years, then last_n_months."""
    years = [y for y in (req.get("years") or []) if isinstance(y, int) and 1 <= y <= 9999]
    if years:
        return dt.date(min(years), 1, 1), dt.date(max(years), 12, 31)
    if req.get("last_n_years"):
        return _shift_years(today, int(req["last_n_years"])), today
    if req.get("last_n_months"):
        return _shift_months(today, int(req["last_n_months"])), today
    return None, None


_AGE_RE = re.compile(r"\bage\s*[:=]?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years|yrs|y/?o)\b",
                     re.IGNORECASE)


def _matches_type(term: str, *fields: str | None) -> bool:
    """True if every query word in `term` fuzzily matches some word across the
    doc's type/name/classification. Reuses the structured-query matcher so
    'lipid' hits 'Lipid Profile' and spelling drift is tolerated."""
    words = _query_words(term)
    if not words:
        return True
    hay = re.findall(r"[a-z0-9]+", " ".join(f.lower() for f in fields if f))
    if not hay:
        return False
    return all(any(_word_score(w, h) >= _FUZZ for h in hay) for w in words)


def _in_window(date_str: str | None, lo: dt.date | None, hi: dt.date | None) -> bool:
    if lo is None and hi is None:
        return True
    d = parse_doc_date(date_str)
    if d is None:
        return False              # undated rows are excluded once a window is set
    return (lo is None or d >= lo) and (hi is None or d <= hi)


def _wanted_type(doc_types: list[str], *fields: str | None) -> bool:
    if not doc_types:
        return True
    return any(_matches_type(t, *fields) for t in doc_types)


def _recent_age(db: Session, patient_id: int, docs: list[dict]) -> int | None:
    """Most recent age = age parsed from the newest in-window document's OCR text;
    fall back to Patient.age. Filesystem/upload timestamps are never used."""
    from app.models import Document
    for doc in sorted(docs, key=lambda d: d.get("report_date") or d.get("date") or "",
                      reverse=True):
        row = db.get(Document, doc["id"])
        m = _AGE_RE.search(row.raw_ocr_text or "") if row else None
        if m:
            return int(m.group(1) or m.group(2))
    p = db.get(Patient, patient_id)
    return p.age if p else None


def gather(db: Session, patient_id: int, doc_types: list[str],
           date_from: dt.date | None, date_to: dt.date | None) -> dict:
    """Aggregate everything the PDF needs for one patient + timeframe + types."""
    docs = [d for d in bsvc.list_documents_timeline(db, patient_id=patient_id)
            if _in_window(d.get("report_date") or d.get("date"), date_from, date_to)
            and _wanted_type(doc_types, d.get("type"), d.get("original_name"),
                             d.get("classification"))]
    kept_ids = {d["id"] for d in docs}

    def _ents(kind: str) -> list[dict]:
        rows = [r for r in bsvc.list_entity_links(db, kind, patient_id=patient_id)
                if r.get("document_id") in kept_ids]
        seen, out = set(), []
        for r in reversed(rows):             # rows are newest-first; reverse for chronology
            key = (r["name"] or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(r)
        return out

    diseases = _ents("disease")
    symptoms = _ents("symptom")
    medications = _ents("medication")
    tests = [r for r in bsvc.list_test_results(db, patient_id=patient_id)
             if r.get("document_id") in kept_ids]
    timeline = sorted(docs, key=lambda d: d.get("report_date") or d.get("date") or "")
    attachments = [{"document_id": d["id"], "name": d.get("original_name") or f"document-{d['id']}",
                    "date": d.get("report_date") or d.get("date"), "file_path": d.get("file"),
                    "type": d.get("type")}
                   for d in timeline if d.get("file")]
    patient = db.get(Patient, patient_id)
    return {
        "patient_id": patient_id,
        "patient_name": patient.name if patient else "Unknown",
        "gender": patient.gender if patient else None,
        "age": _recent_age(db, patient_id, docs),
        "documents": docs,
        "diseases": diseases,
        "symptoms": symptoms,
        "medications": medications,
        "tests": tests,
        "timeline": timeline,
        "attachments": attachments,
    }
