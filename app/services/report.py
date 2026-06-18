"""Aggregate already-extracted DB data into a report payload, plus parse the
NL PDF request and resolve its timeframe. Pure functions over a DB session — no
HTTP, no rendering, no LLM (except parse_request, which takes an injected chat
client). This is the single data source of truth for the PDF pipeline."""
from __future__ import annotations

import datetime as dt
import re  # noqa: F401
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session  # noqa: F401

from app.agent.nodes.structured import _FUZZ, _query_words, _word_score  # noqa: F401
from app.models import Patient  # noqa: F401
from app.services import browse as bsvc  # noqa: F401
from app.services.dates import parse_doc_date  # noqa: F401


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
    years = req.get("years") or []
    if years:
        return dt.date(min(years), 1, 1), dt.date(max(years), 12, 31)
    if req.get("last_n_years"):
        return _shift_years(today, int(req["last_n_years"])), today
    if req.get("last_n_months"):
        return _shift_months(today, int(req["last_n_months"])), today
    return None, None
