"""Split one uploaded file's pages into separate medical reports.

A single scan often bundles several reports (haematology + lipid + X-ray, a
prescription, a discharge summary…), each with its own header, date and tests.
An LLM does the splitting — it groups pages into reports and names them, handling
anything (not just documents literally titled "… REPORT"). A cheap regex is kept
only as a fallback if the LLM call fails. Each report becomes its own document.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

# ---- LLM split schema ----

class ReportSpec(BaseModel):
    title: str = "Medical Report"      # human name, e.g. "Haematological Report", "Chest X-Ray", "Prescription"
    doc_type: str = "document"         # category: lab report | imaging | prescription | discharge | document
    category: str | None = None        # printed department/panel, e.g. Hematology, Biochemistry, X-Ray, Urine
    pages: list[int] = Field(default_factory=list)  # 0-based page indices in this report
    date: str | None = None            # report/collection date if visible


class _ReportSplit(BaseModel):
    reports: list[ReportSpec] = Field(default_factory=list)


_SPLIT_PROMPT = """You are given a scanned medical file, one page per [[PAGE n]] block.
A file may bundle SEVERAL distinct reports (e.g. a haematology report, a lipid panel,
an X-ray, a prescription) — each typically has its own header, date and result table.

Identify each distinct report. For each, return:
- title: a short human name (e.g. "Haematological Report", "Chest X-Ray", "Lipid Profile", "Prescription")
- doc_type: one of lab report, imaging, prescription, discharge, document
- category: the report's printed department/section/panel name if shown (e.g. "Hematology", "Biochemistry", "X-Ray", "Ultrasound", "Urine"), else null
- pages: the 0-based page numbers that belong to it (a report may span pages)
- date: the report/collection/sample date if visible, else null

If the whole file is a single report, return exactly one. Do not invent reports.

{text}"""

# ---- regex fallback ----

_TITLE_RE = re.compile(r"\b([A-Z][A-Z0-9\-/ ]{2,38}?REPORT)\b")
_IMAGING = ("x-ray", "xray", "x ray", "imaging", "ultrasound", "usg", "ct ", "mri",
            "sonogram", "radiolog", "skiagram", "ecg", "echo")


def _clean_title(raw: str) -> str:
    return re.sub(r"\s+", " ", raw).strip().title()


def detect_title(text: str) -> str | None:
    m = _TITLE_RE.search(text)
    return _clean_title(m.group(1)) if m else None


def doc_type_for(title: str | None) -> str:
    if not title:
        return "document"
    t = title.lower()
    if any(k in t for k in _IMAGING):
        return "imaging"
    if "prescription" in t:
        return "prescription"
    if "discharge" in t:
        return "discharge"
    return "lab report" if "report" in t else "document"


# A prescription rarely prints a "… REPORT" header, so detect_title misses it.
# These markers (Rx symbol, dosage-form abbreviations, doctor credentials) catch a
# headerless prescription page so it isn't filed as a generic "document".
_RX_MARKERS = re.compile(
    r"\bR[x×]\b|\bTab\.|\bCap\.|\bSyp\.|\bInj\.|\bMBBS\b|\bFCPS\b|\bDr\.", re.I)


def guess_title_type(text: str) -> tuple[str | None, str]:
    """Best-effort (title, doc_type) for a page with no LLM-assigned report."""
    title = detect_title(text)
    if title:
        return title, doc_type_for(title)
    if _RX_MARKERS.search(text or ""):
        return "Prescription", "prescription"
    return None, "document"


def _regex_segments(pages: list[str]) -> list[dict]:
    """Fallback: start a new report whenever a page shows a '… REPORT' header;
    headerless pages continue the current report."""
    segs: list[dict] = []
    for i, page in enumerate(pages):
        title = detect_title(page)
        if title or not segs:
            segs.append({"title": title, "doc_type": doc_type_for(title),
                         "category": None,
                         "date": None, "text": page, "pages": [i]})
        else:
            segs[-1]["text"] += "\n\n" + page
            segs[-1]["pages"].append(i)
    return segs


def split_reports(chat, pages: list[str]) -> list[dict]:
    """Return [{title, doc_type, date, text, pages}] — one per detected report.
    Single page -> one report (no LLM call). Otherwise ask the LLM; on any failure
    or empty result, fall back to regex header detection."""
    if len(pages) <= 1:
        text = pages[0] if pages else ""
        return [{"title": None, "doc_type": "document", "category": None,
                 "date": None, "text": text, "pages": [0]}]

    marked = "\n\n".join(f"[[PAGE {i}]]\n{p}" for i, p in enumerate(pages))
    reports: list[ReportSpec] = []
    try:
        split = chat.structured(_SPLIT_PROMPT.format(text=marked), _ReportSplit)
        reports = split.reports
    except Exception:  # noqa: BLE001 — LLM/parse failure -> regex fallback
        reports = []

    if not reports:
        return _regex_segments(pages)

    n = len(pages)
    specs = [(r, sorted({i for i in r.pages if 0 <= i < n})) for r in reports]

    # The LLM often names N reports but leaves `pages` empty, or assigns the same
    # page to several. Defaulting each to "all pages" would slice every report to
    # the whole bundle, so every card opens page 1. When the split is degenerate
    # (any report has no pages, or pages overlap) fall back to regex header-split,
    # which gives each titled report its own page(s).
    assigned = [i for _, idxs in specs for i in idxs]
    degenerate = any(not idxs for _, idxs in specs) or len(assigned) != len(set(assigned))
    if len(specs) > 1 and degenerate:
        # One title per page (N reports, N pages) -> trust the titles, assign pages
        # 1:1 in scan order. Otherwise the page split is unreliable -> regex headers.
        if len(specs) == n:
            specs = [(r, [i]) for i, (r, _) in enumerate(specs)]
        else:
            return _regex_segments(pages)

    out: list[dict] = []
    for r, idxs in specs:
        idxs = idxs or list(range(n))  # single-report case only
        out.append({
            "title": (r.title or "").strip() or None,
            "doc_type": (r.doc_type or "document").strip().lower(),
            "category": (r.category or "").strip() or None,
            "date": r.date,
            "text": "\n\n".join(pages[i] for i in idxs),
            "pages": idxs,
        })
    # The LLM sometimes ignores a page entirely (e.g. a prescription it didn't
    # recognise). A page with real text must never vanish — give each orphan its
    # own report so it still gets ingested.
    covered = {i for s in out for i in s["pages"]}
    for i in range(n):
        if i in covered or not (pages[i] or "").strip():
            continue
        title, dtype = guess_title_type(pages[i])
        out.append({"title": title, "doc_type": dtype, "category": None,
                    "date": None, "text": pages[i], "pages": [i]})
    out.sort(key=lambda s: s["pages"][0])
    return out
