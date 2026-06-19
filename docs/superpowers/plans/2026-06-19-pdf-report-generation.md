# PDF Report Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a downloadable, sectioned medical PDF from a natural-language request, reusing already-extracted DB data, with two human approval gates (plan + delivery).

**Architecture:** New `generate_pdf` intent → 4 LangGraph nodes (`plan_report` → `confirm_report` gate → `build_report` → `deliver_report` gate). Pure services aggregate DB data (`report.py`), render matplotlib charts (`charts.py`), and assemble the PDF with pymupdf (`pdf.py`). Frontend reuses the existing interrupt-card + SSE-stepper machinery.

**Tech Stack:** Python, FastAPI, LangGraph, SQLAlchemy, pymupdf (fitz), matplotlib, vanilla TypeScript.

---

## Conventions

- Run tests with the test DB: tests rely on `TEST_DATABASE_URL` (distinct from `DATABASE_URL`) — the pgvector docker on `:5433`. `conftest.py` refuses to run otherwise.
- Run a single test file: `python -m pytest tests/<file> -v`.
- Node tests build a `Deps` dataclass with fakes (see `tests/agent/test_router.py`).
- Commit after each task passes.

---

## Task 1: Add matplotlib dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add the dependency**

Append to `requirements.txt` (after `pillow>=10.0`):

```
matplotlib>=3.8
```

- [ ] **Step 2: Install**

Run: `pip install 'matplotlib>=3.8'`
Expected: installs matplotlib + its deps (numpy already present via pgvector? if not it installs).

- [ ] **Step 3: Verify headless import works**

Run: `python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; print('ok')"`
Expected: prints `ok` (no display/backend error).

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "build(pdf): add matplotlib for chart rendering"
```

---

## Task 2: report.py — request parsing + timeframe resolution (pure)

**Files:**
- Create: `app/services/report.py`
- Test: `tests/test_report_service.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report_service.py
import datetime as dt

from app.services import report


def test_resolve_timeframe_explicit_years():
    req = {"years": [2021, 2022]}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2021, 1, 1)
    assert hi == dt.date(2022, 12, 31)


def test_resolve_timeframe_last_n_years():
    req = {"last_n_years": 3}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2023, 6, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_last_n_months():
    req = {"last_n_months": 4}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2026, 2, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_months_cross_year():
    req = {"last_n_months": 8}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 3, 10))
    assert lo == dt.date(2025, 7, 10)
    assert hi == dt.date(2026, 3, 10)


def test_resolve_timeframe_none_is_all_time():
    assert report.resolve_timeframe({}, dt.date(2026, 6, 19)) == (None, None)


def test_resolve_timeframe_leap_day_guard():
    # last_n_years landing on Feb 29 of a non-leap year clamps to Feb 28.
    lo, hi = report.resolve_timeframe({"last_n_years": 1}, dt.date(2024, 2, 29))
    assert lo == dt.date(2023, 2, 28)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_report_service.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'app.services.report' has no attribute 'resolve_timeframe'`.

- [ ] **Step 3: Write the implementation**

```python
# app/services/report.py
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
    years = req.get("years") or []
    if years:
        return dt.date(min(years), 1, 1), dt.date(max(years), 12, 31)
    if req.get("last_n_years"):
        return _shift_years(today, int(req["last_n_years"])), today
    if req.get("last_n_months"):
        return _shift_months(today, int(req["last_n_months"])), today
    return None, None
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_report_service.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add app/services/report.py tests/test_report_service.py
git commit -m "feat(pdf): parse PDF request + resolve timeframe"
```

---

## Task 3: report.py — gather() aggregation + age + doc filtering

**Files:**
- Modify: `app/services/report.py`
- Test: `tests/test_report_service.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_report_service.py
from app.db import SessionLocal
from app.models import (
    Disease, Document, DocumentEntity, MedicalTest, Patient, TestResult,
)


def _doc(db, patient, doc_date, doc_type, raw=""):
    d = Document(patient_id=patient.id, doc_type=doc_type, report_date=doc_date,
                 raw_ocr_text=raw, original_name=f"{doc_type}.pdf")
    db.add(d); db.flush()
    return d


def _disease(db, doc, name):
    dis = Disease(name=name); db.add(dis); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="disease", entity_id=dis.id))


def _result(db, doc, test_name, value, unit, ref):
    mt = MedicalTest(name=test_name); db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value=value, unit=unit, reference_range=ref)
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result", entity_id=tr.id))


def test_gather_filters_to_window_and_doc_type():
    db = SessionLocal()
    try:
        p = Patient(name="Gather One", age=40); db.add(p); db.flush()
        d_in = _doc(db, p, dt.date(2022, 5, 1), "lipid profile", raw="Age: 55 years")
        _result(db, d_in, "LDL", "130", "mg/dL", "0-100")
        d_old = _doc(db, p, dt.date(2018, 1, 1), "lipid profile")   # before window
        _result(db, d_old, "LDL", "90", "mg/dL", "0-100")
        d_other = _doc(db, p, dt.date(2022, 6, 1), "x-ray")          # wrong type
        db.commit()

        data = report.gather(db, p.id, ["lipid profile"],
                             dt.date(2021, 1, 1), dt.date(2023, 1, 1))
        names = {doc["original_name"] for doc in data["documents"]}
        assert names == {"lipid profile.pdf"}          # only the in-window lipid doc
        assert any(t["test"] == "LDL" and t["value"] == "130" for t in data["tests"])
        assert all(t["value"] != "90" for t in data["tests"])  # old doc excluded
    finally:
        db.close()


def test_gather_most_recent_age_from_newest_doc_ocr():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Age", age=40); db.add(p); db.flush()
        _doc(db, p, dt.date(2020, 1, 1), "lab report", raw="Age: 50 years")
        _doc(db, p, dt.date(2023, 1, 1), "lab report", raw="Age : 53 yrs")  # newest
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert data["age"] == 53      # newest doc's OCR age, not Patient.age=40
    finally:
        db.close()


def test_gather_age_falls_back_to_patient_age():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Fallback", age=72); db.add(p); db.flush()
        _doc(db, p, dt.date(2023, 1, 1), "lab report", raw="no age here")
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert data["age"] == 72
    finally:
        db.close()


def test_gather_dedupes_diseases_preserving_order():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Dx"); db.add(p); db.flush()
        d1 = _doc(db, p, dt.date(2021, 1, 1), "note"); _disease(db, d1, "Anemia")
        d2 = _doc(db, p, dt.date(2022, 1, 1), "note"); _disease(db, d2, "Anemia")
        d3 = _doc(db, p, dt.date(2022, 6, 1), "note"); _disease(db, d3, "Diabetes")
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert [x["name"] for x in data["diseases"]] == ["Anemia", "Diabetes"]
    finally:
        db.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_report_service.py -k gather -v`
Expected: FAIL — `AttributeError: module 'app.services.report' has no attribute 'gather'`.

- [ ] **Step 3: Write the implementation (append to `app/services/report.py`)**

```python
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
    for doc in sorted(docs, key=lambda d: d.get("report_date") or d.get("date") or "",
                      reverse=True):
        from app.models import Document
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
```

Note: remove the empty `for r in rows: pass` stub — it's a placeholder shown for clarity; the real loop is `for r in reversed(rows)`. Delete the no-op loop when implementing.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_report_service.py -v`
Expected: PASS (all tests, including the 6 from Task 2).

- [ ] **Step 5: Commit**

```bash
git add app/services/report.py tests/test_report_service.py
git commit -m "feat(pdf): gather aggregates DB data with window + type + age"
```

---

## Task 4: charts.py — matplotlib trend chart

**Files:**
- Create: `app/services/charts.py`
- Test: `tests/test_charts_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_charts_service.py
from app.services import charts


def _series():
    return {
        "key": "ldl", "label": "LDL", "unit": "mg/dL",
        "ref_low": 0.0, "ref_high": 100.0,
        "points": [
            {"date": "2021-01-01", "value": 90.0, "in_range": True},
            {"date": "2022-01-01", "value": 130.0, "in_range": False},
        ],
    }


def test_render_metric_chart_returns_png_bytes():
    png = charts.render_metric_chart(_series())
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"     # PNG magic header
    assert len(png) > 1000                       # a real image, not an empty buffer


def test_render_metric_chart_without_reference_band():
    s = _series(); s["ref_low"] = None; s["ref_high"] = None
    png = charts.render_metric_chart(s)          # must not raise without a band
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_charts_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.charts'`.

- [ ] **Step 3: Write the implementation**

```python
# app/services/charts.py
"""Render a single numeric trend metric to a PNG (bytes) for the PDF. Feeds off
trends.metric_series output. Headless Agg backend — locked before pyplot import
so it never tries to open a display."""
from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def render_metric_chart(series: dict) -> bytes:
    points = series.get("points") or []
    dates = [p["date"] for p in points]
    values = [p["value"] for p in points]
    unit = series.get("unit") or ""
    label = series.get("label") or series.get("key") or "Metric"

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(dates, values, marker="o", linewidth=1.6)
    lo, hi = series.get("ref_low"), series.get("ref_high")
    if lo is not None and hi is not None:
        ax.axhspan(lo, hi, color="tab:green", alpha=0.12, label="reference range")
        ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{label} over time")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"{label} ({unit})" if unit else label)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_charts_service.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add app/services/charts.py tests/test_charts_service.py
git commit -m "feat(pdf): matplotlib trend chart -> PNG bytes"
```

---

## Task 5: pdf.py — assemble report + append attachments

**Files:**
- Create: `app/services/pdf.py`
- Test: `tests/test_pdf_service.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdf_service.py
import fitz   # pymupdf

from app.services import pdf


def _data():
    return {
        "patient_name": "Jane Doe", "age": 55, "gender": "F",
        "timeframe_label": "2021–2023",
        "diseases": [{"name": "Hyperlipidemia", "date": "2022-05-01"}],
        "symptoms": [],
        "tests": [{"test": "LDL", "value": "130", "unit": "mg/dL",
                   "reference_range": "0-100", "date": "2022-05-01",
                   "doc_type": "lipid profile", "source": "LDL 130"}],
        "timeline": [{"original_name": "lipid.pdf", "type": "lipid profile",
                      "report_date": "2022-05-01"}],
    }


def _tiny_pdf_bytes() -> bytes:
    d = fitz.open()
    d.new_page()
    return d.tobytes()


def test_build_report_returns_valid_pdf():
    out = pdf.build_report(_data(), charts=[], attachments=[])
    doc = fitz.open("pdf", out)
    assert doc.page_count >= 1
    text = "".join(p.get_text() for p in doc)
    assert "Jane Doe" in text
    assert "LDL" in text


def test_build_report_appends_pdf_attachment(tmp_path):
    src = tmp_path / "orig.pdf"
    src.write_bytes(_tiny_pdf_bytes())   # 1-page source PDF
    body_only = fitz.open("pdf", pdf.build_report(_data(), [], [])).page_count
    out = pdf.build_report(_data(), charts=[],
                           attachments=[{"name": "orig.pdf", "date": "2022-05-01",
                                         "file_path": str(src), "type": "lipid profile"}])
    doc = fitz.open("pdf", out)
    # appendix index page + the appended source page => at least body + 2
    assert doc.page_count >= body_only + 2


def test_build_report_chart_adds_page():
    png = fitz.open()  # build a trivial 1x1 png via matplotlib-free route
    # use a real PNG produced by charts to stay representative
    from app.services import charts
    chart_png = charts.render_metric_chart({
        "label": "LDL", "unit": "mg/dL", "ref_low": 0.0, "ref_high": 100.0,
        "points": [{"date": "2021-01-01", "value": 90.0},
                   {"date": "2022-01-01", "value": 130.0}]})
    body_only = fitz.open("pdf", pdf.build_report(_data(), [], [])).page_count
    out = pdf.build_report(_data(), charts=[("LDL over time", chart_png)], attachments=[])
    assert fitz.open("pdf", out).page_count >= body_only + 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pdf_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.pdf'`.

- [ ] **Step 3: Write the implementation**

```python
# app/services/pdf.py
"""Assemble the final report PDF with pymupdf (fitz): reflowable HTML body via
Story, matplotlib chart pages, then appended original source documents with an
appendix index. Returns PDF bytes. No new heavy PDF dependency."""
from __future__ import annotations

import html as _html
import io
import os

import fitz

_A4 = fitz.paper_rect("a4")
_MARGIN = (40, 40, -40, -40)


def _esc(v) -> str:
    return _html.escape(str(v)) if v is not None else ""


def _section(title: str, inner: str) -> str:
    return f"<h2>{_esc(title)}</h2>{inner}" if inner else ""


def _report_html(data: dict) -> str:
    name = _esc(data.get("patient_name"))
    age = data.get("age")
    gender = _esc(data.get("gender"))
    tf = _esc(data.get("timeframe_label") or "All records")

    cover = (f"<div style='text-align:center'>"
             f"<h1 style='font-size:26pt'>Medical Report</h1>"
             f"<h1 style='font-size:20pt;color:#444'>{name}</h1>"
             f"<p>Timeframe: {tf}</p></div>")

    info_rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{_esc(v)}</td></tr>"
        for k, v in [("Name", data.get("patient_name")),
                     ("Age", age if age is not None else "—"),
                     ("Gender", gender or "—")])
    info = _section("Patient Information", f"<table>{info_rows}</table>")

    def _named(items):
        if not items:
            return ""
        return "<ul>" + "".join(
            f"<li>{_esc(i['name'])}"
            f"{(' · ' + _esc(i.get('date'))) if i.get('date') else ''}</li>"
            for i in items) + "</ul>"

    diseases = _section("Disease Summary", _named(data.get("diseases")))
    symptoms = _section("Symptoms Summary", _named(data.get("symptoms")))

    tests = data.get("tests") or []
    if tests:
        head = ("<tr><th>Test</th><th>Value</th><th>Unit</th><th>Reference</th>"
                "<th>Date</th><th>Source</th></tr>")
        body = "".join(
            f"<tr><td>{_esc(t.get('test'))}</td><td>{_esc(t.get('value'))}</td>"
            f"<td>{_esc(t.get('unit'))}</td><td>{_esc(t.get('reference_range'))}</td>"
            f"<td>{_esc(t.get('date'))}</td><td>{_esc(t.get('doc_type'))}</td></tr>"
            for t in tests)
        tests_html = _section("Medical Test Results", f"<table>{head}{body}</table>")
    else:
        tests_html = ""

    tl = data.get("timeline") or []
    timeline = _section("Timeline of Findings", "<ul>" + "".join(
        f"<li>{_esc(d.get('report_date') or d.get('date'))} — "
        f"{_esc(d.get('original_name'))} ({_esc(d.get('type'))})</li>"
        for d in tl) + "</ul>") if tl else ""

    css = ("<style>body{font-family:sans-serif;font-size:11pt;color:#222}"
           "h2{border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:18px}"
           "table{border-collapse:collapse;width:100%}"
           "td,th{border:1px solid #ddd;padding:4px;text-align:left;font-size:10pt}"
           "</style>")
    return (f"<html><head>{css}</head><body>{cover}{info}{diseases}{symptoms}"
            f"{tests_html}{timeline}</body></html>")


def _render_body(html: str) -> bytes:
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    story = fitz.Story(html=html)
    where = _A4 + _MARGIN
    more = 1
    while more:
        dev = writer.begin_page(_A4)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    return buf.getvalue()


def _appendix_index(doc: fitz.Document, attachments: list[dict]) -> None:
    page = doc.new_page(width=_A4.width, height=_A4.height)
    y = 60
    page.insert_text((40, y), "Attached Original Documents", fontsize=15)
    y += 28
    for i, att in enumerate(attachments, 1):
        line = f"A{i}. {att.get('name')}  ·  {att.get('date') or 'undated'}  ·  {att.get('type') or ''}"
        page.insert_text((40, y), line[:110], fontsize=10)
        y += 18
        if y > _A4.height - 50:
            page = doc.new_page(width=_A4.width, height=_A4.height)
            y = 50


def build_report(data: dict, charts: list[tuple[str, bytes]],
                 attachments: list[dict]) -> bytes:
    """data = gather() output (+ timeframe_label); charts = [(title, png_bytes)];
    attachments = [{name, date, file_path, type}]. Returns assembled PDF bytes."""
    doc = fitz.open("pdf", _render_body(_report_html(data)))

    # Charts & Trends — one page per chart.
    for title, png in charts:
        page = doc.new_page(width=_A4.width, height=_A4.height)
        page.insert_text((40, 45), title, fontsize=13)
        page.insert_image(fitz.Rect(40, 70, _A4.width - 40, 420), stream=png,
                          keep_proportion=True)

    # Attached Original Documents (+ appendix index), ordered by document date.
    if attachments:
        _appendix_index(doc, attachments)
        for att in attachments:
            path = att.get("file_path")
            if not path or not os.path.exists(path):
                continue
            if path.lower().endswith(".pdf"):
                with fitz.open(path) as src:
                    doc.insert_pdf(src)
            else:   # image
                page = doc.new_page(width=_A4.width, height=_A4.height)
                page.insert_image(fitz.Rect(30, 30, _A4.width - 30, _A4.height - 30),
                                  filename=path, keep_proportion=True)
    return doc.tobytes()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_pdf_service.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/services/pdf.py tests/test_pdf_service.py
git commit -m "feat(pdf): assemble sectioned report + appended attachments"
```

---

## Task 6: storage.save_report + download endpoint

**Files:**
- Modify: `app/storage.py`
- Modify: `app/api/routes_chat.py`
- Test: `tests/test_pdf_download.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pdf_download.py
import fitz
from fastapi.testclient import TestClient

from app import storage
from app.api.server import app


def test_save_report_and_download(tmp_path, monkeypatch):
    client = TestClient(app)
    d = fitz.open(); d.new_page()
    path = storage.save_report(d.tobytes())
    assert path.endswith(".pdf")
    name = path.rsplit("/", 1)[-1]
    r = client.get(f"/api/chat/report/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_download_rejects_path_traversal():
    client = TestClient(app)
    r = client.get("/api/chat/report/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)


def test_download_missing_is_404():
    client = TestClient(app)
    r = client.get("/api/chat/report/deadbeef.pdf")
    assert r.status_code == 404
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_pdf_download.py -v`
Expected: FAIL — `AttributeError: module 'app.storage' has no attribute 'save_report'` / 404 routing.

- [ ] **Step 3: Implement storage.save_report**

Append to `app/storage.py`:

```python
def save_report(data: bytes) -> str:
    """Persist a generated report PDF under STORAGE_DIR/_reports/<uuid>.pdf."""
    target = (Path(STORAGE_DIR) / "_reports" / f"{uuid.uuid4().hex}.pdf").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)
```

- [ ] **Step 4: Implement the download route**

Add to `app/api/routes_chat.py` (after the existing imports add `from fastapi.responses import FileResponse` is already imported as StreamingResponse — add FileResponse; and `import os`):

```python
@router.get("/report/{name}")
def report_file(name: str):
    # Serve a generated report by its stored uuid filename. Reject anything that
    # isn't a bare <hex>.pdf name to block path traversal.
    if "/" in name or "\\" in name or ".." in name or not name.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="bad report name")
    base = (Path(storage.STORAGE_DIR) / "_reports").resolve()
    path = (base / name).resolve()
    if not str(path).startswith(str(base)) or not path.exists():
        raise HTTPException(status_code=404, detail="report not found")
    return FileResponse(str(path), media_type="application/pdf",
                        filename="medical-report.pdf")
```

Ensure the import line reads: `from fastapi.responses import FileResponse, StreamingResponse`.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/test_pdf_download.py -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add app/storage.py app/api/routes_chat.py tests/test_pdf_download.py
git commit -m "feat(pdf): store report + secured download endpoint"
```

---

## Task 7: nodes — plan_report + confirm_report (Gate A)

**Files:**
- Create: `app/agent/nodes/report.py`
- Test: `tests/agent/test_report_nodes.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/agent/test_report_nodes.py
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
    # interrupt() returns the resume payload; monkeypatch it to simulate a decision.
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [{"role": "user", "content": "x"}],
             "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {}}
    rmod.interrupt = lambda payload: {"approved": False}        # type: ignore
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "end"
    assert "cancel" in out["messages"][-1]["content"].lower()


def test_confirm_report_modify_replans(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [{"role": "user", "content": "x"}],
             "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {"last_n_years": 3}}
    rmod.interrupt = lambda payload: {"approved": True,
                                      "modify": {"last_n_years": 5}}   # type: ignore
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "replan"
    assert out["report_request"]["last_n_years"] == 5


def test_confirm_report_approve_builds(db_session_factory):
    import app.agent.nodes.report as rmod
    sf = db_session_factory
    state = {"messages": [], "report_plan": {"patient_id": 1, "documents": []},
             "report_request": {}}
    rmod.interrupt = lambda payload: {"approved": True}            # type: ignore
    out = rmod.confirm_report_node(state, _cfg(sf=sf))
    assert out["report_decision"] == "build"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/agent/test_report_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.nodes.report'`.

- [ ] **Step 3: Write the implementation**

```python
# app/agent/nodes/report.py
"""PDF report generation nodes. Two HITL gates: confirm_report (plan) and
deliver_report (delivery). Between them, build_report runs automatically."""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

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
    return f"{lo.isoformat() if lo else '…'} – {hi.isoformat() if hi else '…'}"


def plan_report_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    # Reuse an existing parsed request on a re-plan; else parse the NL text once.
    req = state.get("report_request")
    if req is None:
        req = report_svc.parse_request(deps.chat, _last_user_text(state)).model_dump()
    today = dt.date.today()
    lo, hi = report_svc.resolve_timeframe(req, today)
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
        return _say(state, "Report cancelled — nothing generated.", report_decision="end")
    mods = decision.get("modify")
    if mods:
        return {"report_decision": "replan",
                "report_request": {**(state.get("report_request") or {}), **mods},
                "report_plan": None}
    return {"report_decision": "build"}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/agent/test_report_nodes.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes/report.py tests/agent/test_report_nodes.py
git commit -m "feat(pdf): plan_report + confirm_report gate (Gate A)"
```

---

## Task 8: nodes — build_report + deliver_report (Gate B)

**Files:**
- Modify: `app/agent/nodes/report.py`
- Test: `tests/agent/test_report_nodes.py` (append)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/agent/test_report_nodes.py
import fitz


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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/agent/test_report_nodes.py -k "build_report_writes or deliver" -v`
Expected: FAIL — `AttributeError: ... has no attribute 'build_report_node'`.

- [ ] **Step 3: Write the implementation (append to `app/agent/nodes/report.py`)**

```python
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
                            "page_count": len(pdf_svc.fitz.open("pdf", pdf_bytes))}}


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
```

Note: `pdf_svc.fitz` references the `fitz` already imported inside `app/services/pdf.py`. If lint flags it, instead add `import fitz` at the top of `app/agent/nodes/report.py` and use `fitz.open(...)`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/agent/test_report_nodes.py -v`
Expected: PASS (8 tests total).

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes/report.py tests/agent/test_report_nodes.py
git commit -m "feat(pdf): build_report + deliver_report gate (Gate B)"
```

---

## Task 9: router + graph wiring + node labels

**Files:**
- Modify: `app/agent/router.py`
- Modify: `app/agent/graph.py`
- Modify: `app/agent/state.py`
- Modify: `app/api/runtime.py`
- Test: `tests/agent/test_router.py` (append), `tests/agent/test_graph.py` (append), `tests/test_api_chat.py` (extend label test)

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/agent/test_router.py
def test_router_generate_pdf():
    state = {"messages": [{"role": "user", "content": "make a pdf of Jane's lipid results"}]}
    out = classify_intent(state, _cfg(_FakeChat("generate_pdf")))
    assert out["intent"] == "generate_pdf"
```

```python
# append to tests/agent/test_graph.py  (match the existing import/build style there)
def test_graph_has_report_nodes():
    from app.agent.graph import build_graph
    g = build_graph()
    nodes = set(g.get_graph().nodes)
    for n in ["plan_report", "confirm_report", "build_report", "deliver_report"]:
        assert n in nodes
```

```python
# extend tests/test_api_chat.py::test_node_labels_cover_key_nodes
# add the four report nodes to the asserted list:
#   for node in ["router", "extract_text", "generate_answer",
#                "plan_report", "build_report", "deliver_report"]:
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/agent/test_router.py::test_router_generate_pdf tests/agent/test_graph.py::test_graph_has_report_nodes -v`
Expected: FAIL — intent falls through to `rag_query`; nodes missing from graph.

- [ ] **Step 3a: Update the router**

In `app/agent/router.py`, update `_VALID` and `_PROMPT`, and add the branch:

```python
_VALID = {"ingest", "structured_query", "rag_query", "edit", "generate_pdf"}
```

Add to `_PROMPT` (above the `structured_query` line):

```
- generate_pdf: asking to MAKE/CREATE/GENERATE a PDF/report/document OUT OF records (e.g. "make a pdf of all lipid results of patient X for the last 3 years", "make a pdf from all reports of Bob").
```

In `classify_intent`, add before the `structured` check:

```python
    if "generate_pdf" in label or "pdf" in label:
        return {"intent": "generate_pdf"}
```

- [ ] **Step 3b: Add state keys**

In `app/agent/state.py`, inside `AgentState`, add under the `# query` group:

```python
    report_request: dict[str, Any] | None
    report_plan: dict[str, Any] | None
    report_decision: str | None
    report_path: str | None
    report_url: str | None
```

- [ ] **Step 3c: Wire the graph**

In `app/agent/graph.py`:

Add the import:

```python
from app.agent.nodes.report import (
    build_report_node, confirm_report_node, deliver_report_node, plan_report_node,
)
```

Register nodes (after the edit nodes block):

```python
    # pdf report
    g.add_node("plan_report", plan_report_node)
    g.add_node("confirm_report", confirm_report_node)
    g.add_node("build_report", build_report_node)
    g.add_node("deliver_report", deliver_report_node)
```

Add `"generate_pdf": "plan_report"` to the `router` conditional-edges map:

```python
    g.add_conditional_edges("router", _route, {
        "ingest": "dedup_check",
        "structured_query": "parse_filters",
        "rag_query": "require_patient",
        "edit": "plan_edit",
        "generate_pdf": "plan_report",
    })
```

Add the report chain (near the edit chain):

```python
    # pdf report chain — plan -> Gate A -> build -> Gate B -> (download | regenerate)
    g.add_conditional_edges("plan_report",
                            lambda s: "confirm" if s.get("report_plan") else "end",
                            {"confirm": "confirm_report", "end": END})
    g.add_conditional_edges("confirm_report",
                            lambda s: s.get("report_decision") or "end",
                            {"build": "build_report", "replan": "plan_report",
                             "end": END})
    g.add_edge("build_report", "deliver_report")
    g.add_conditional_edges("deliver_report",
                            lambda s: "rebuild" if s.get("report_decision") == "rebuild"
                            else "end",
                            {"rebuild": "build_report", "end": END})
```

- [ ] **Step 3d: Add node labels**

In `app/api/runtime.py`, add to `NODE_LABELS`:

```python
    "plan_report": "🧾 Planning your report…",
    "confirm_report": "✅ Awaiting your approval…",
    "build_report": "📄 Building the PDF…",
    "deliver_report": "📦 Finalizing your report…",
```

- [ ] **Step 4: Run the full suite**

Run: `python -m pytest tests/agent/test_router.py tests/agent/test_graph.py tests/test_api_chat.py -v`
Expected: PASS, including the new router/graph/label assertions.

- [ ] **Step 5: Commit**

```bash
git add app/agent/router.py app/agent/graph.py app/agent/state.py app/api/runtime.py \
        tests/agent/test_router.py tests/agent/test_graph.py tests/test_api_chat.py
git commit -m "feat(pdf): route generate_pdf intent through report chain"
```

---

## Task 10: frontend — interrupt cards for both gates

**Files:**
- Modify: `medagentic-dashboard/src/main.ts`

No unit test (the project has no JS test runner); verification is `tsc` + a manual run note.

- [ ] **Step 1: Add the two interrupt cards**

In `interruptCardHtml(payload, idx)` (after the `confirm_edit` branch, before the fallback), add:

```typescript
  if (payload.type === 'confirm_report') {
    const p = payload.plan || {};
    const docs = (p.documents || []).map((d: any) =>
      `<li>${esc(d.name)} · ${esc(d.type || 'document')}${d.date ? ' · ' + esc(d.date) : ''}</li>`).join('');
    return `
      <div class="rounded-xl border border-[#E5E2DC] bg-white p-4 text-sm">
        <p class="font-semibold mb-1">Report plan</p>
        <p>Patient: <b>${esc(p.patient_name)}</b></p>
        <p>Timeframe: ${esc(p.timeframe_label)}</p>
        <p class="mt-1">${(p.counts?.documents ?? 0)} document(s):</p>
        <ul class="list-disc ml-5 my-1">${docs}</ul>
        <div class="flex gap-2 mt-3">
          <button data-idx="${idx}" data-act="confirm" class="hitl px-3 py-1.5 rounded bg-[#1A1A1A] text-white">Approve</button>
          <button data-idx="${idx}" data-act="cancel" class="hitl px-3 py-1.5 rounded border">Cancel</button>
        </div>
      </div>`;
  }
  if (payload.type === 'confirm_delivery') {
    const s = payload.summary || {};
    return `
      <div class="rounded-xl border border-[#E5E2DC] bg-white p-4 text-sm">
        <p class="font-semibold mb-1">Report ready</p>
        <p>${s.page_count ?? '?'} pages · ${s.chart_count ?? 0} chart(s) · ${s.attachment_count ?? 0} attachment(s)</p>
        <div class="flex gap-2 mt-3">
          <a href="${esc(s.url)}" target="_blank" class="px-3 py-1.5 rounded bg-[#1A1A1A] text-white">Download</a>
          <button data-idx="${idx}" data-act="regenerate" class="hitl px-3 py-1.5 rounded border">Regenerate</button>
        </div>
      </div>`;
  }
```

- [ ] **Step 2: Handle the resume actions**

In the `.hitl` click handler (around line 845, where `confirm_ingest`/`confirm_edit` are handled), add branches:

```typescript
      } else if (payload.type === 'confirm_report') {
        resume = t.dataset.act === 'confirm' ? { approved: true } : { approved: false };
      } else if (payload.type === 'confirm_delivery') {
        resume = t.dataset.act === 'regenerate' ? { regenerate: true } : { approved: true };
      }
```

For `confirm_delivery`, show the build stepper while regenerating:

```typescript
      runResume(resume, payload.type === 'confirm_delivery' && t.dataset.act === 'regenerate');
```

(Match the existing `runResume(resume, …)` call signature already used for `confirm_ingest`.)

- [ ] **Step 3: Type-check**

Run: `cd medagentic-dashboard && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 4: Manual smoke (note for reviewer)**

Start backend (`uvicorn app.api.server:app --reload`) and the frontend (`cd medagentic-dashboard && npm run dev`). Select a patient with several dated lab docs, send "make a pdf of all lab reports for the last 3 years". Confirm: Gate A card lists patient + docs → Approve → stepper → Gate B card with Download/Regenerate → Download opens the PDF.

- [ ] **Step 5: Commit**

```bash
git add medagentic-dashboard/src/main.ts
git commit -m "feat(pdf): Gate A/B interrupt cards + download in chat UI"
```

---

## Task 11: Full regression + branch wrap-up

- [ ] **Step 1: Run the whole suite**

Run: `python -m pytest -q`
Expected: all green except the pre-existing known-broken tests recorded in project memory (do not "fix" unrelated failures; confirm they predate this branch with `git stash` + compare if unsure).

- [ ] **Step 2: Confirm no stray files / debug prints**

Run: `git status` and `git diff --stat origin/master...HEAD`
Expected: only the files this plan touched.

- [ ] **Step 3: Push the branch (do NOT merge — code review pending)**

```bash
git push -u origin feat/pdf-generation
```

---

## Self-review notes (coverage vs spec)

- Cover/patient/disease/symptom/tests/timeline/attachments sections → Task 5 (`_report_html`, `_appendix_index`, `build_report`).
- Charts (matplotlib, units, reference band, chronological) → Task 4 + Task 8 window filtering.
- Most-recent age by document date → Task 3 `_recent_age`.
- Timeframe (years / last-N-years / last-N-months / all) → Task 2 `resolve_timeframe`.
- Doc-type filtering with spelling tolerance → Task 3 `_matches_type` (reuses structured matcher).
- Two HITL gates + modify/regenerate recovery → Tasks 7–9.
- Download endpoint (path-traversal-safe) → Task 6.
- Provenance / no-fabrication / determinism → reuse of `browse` rows (provenance carried), DB-only data, pure `gather`.
- Out of scope per spec decision: estimated-time, cancel-mid-build, 6-gate flow, re-extraction.
