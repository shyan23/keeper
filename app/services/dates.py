from __future__ import annotations

import datetime as dt
import re

# Try explicit formats first (day-first for dd/mm ambiguity — Bangladeshi lab reports).
_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y", "%d %b, %Y",
    "%Y/%m/%d",
]


def parse_doc_date(s: str | None) -> dt.date | None:
    if not s:
        return None
    text = s.strip()
    for fmt in _FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Last resort: pull a yyyy-mm-dd or dd/mm/yyyy substring out of a longer line.
    m = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", text)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", text)
    if m:
        y = int(m.group(3))
        y += 2000 if y < 100 else 0
        try:
            return dt.date(y, int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return None


# Lines that, when present, strongly indicate a report/collection date follows.
_DATE_HINTS = ("date", "received", "reported", "collected", "sample", "drawn", "printed")


def date_from_text(text: str | None) -> dt.date | None:
    """Fallback when the LLM didn't return a doc_date: pull the report date out of
    the OCR text. Prefer lines that name a date (e.g. 'Sample Received: 30/04/21'),
    then fall back to the first date-like token anywhere in the document."""
    if not text:
        return None
    for line in text.splitlines():
        low = line.lower()
        if any(h in low for h in _DATE_HINTS):
            d = parse_doc_date(line)
            if d:
                return d
    return parse_doc_date(text)
