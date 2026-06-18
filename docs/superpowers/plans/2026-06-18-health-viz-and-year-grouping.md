# Health Visualization + Year-Grouped Records Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add test-result trend charts (Trends tab, Chart.js) and reorganize the All tab into Year → category → documents, with report categories extracted by the LLM.

**Architecture:** A backend trend-series service is the single source of truth for chart data (Chart.js now, matplotlib/PDF later). Report categories are LLM-extracted during segmentation and stored in the existing `Document.classification` column. The All-tab regrouping and the Trends tab are pure frontend changes over new/extended endpoints.

**Tech Stack:** Python / FastAPI / SQLAlchemy / Pydantic (backend, pytest); vanilla TS / Vite / esbuild + Chart.js (frontend, verified by `tsc --noEmit`).

---

## File Structure

**Backend**
- `app/services/documents.py` — `create_document` gains `classification` param (modify).
- `app/services/browse.py` — `list_documents_timeline` returns `classification` (modify).
- `app/api/mapping.py` — `document_to_out` emits `category` (modify).
- `app/api/schemas.py` — `DocumentOut.category`; new `MetricOut`, `SeriesPointOut`, `SeriesOut` (modify).
- `app/services/segment.py` — `ReportSpec.category` + prompt; `split_reports` emits `category` (modify).
- `app/agent/nodes/ingest.py` — pass `seg["category"]` into `create_document` (modify).
- `app/services/trends.py` — **new** — `list_metrics`, `metric_series` (the shared series source of truth).
- `app/api/routes_browse.py` — two new trend routes (modify).

**Frontend** (`medagentic-dashboard/`)
- `src/types.ts` — `ApiDocument.category`; new `TrendMetric`, `TrendSeries`, `TrendPoint` (modify).
- `src/api.ts` — `getTrendMetrics`, `getTrendSeries` (modify).
- `src/grouping.ts` — **new** — pure `groupDocsByYear` (so the bucket logic is isolated and type-checkable).
- `src/main.ts` — rewrite `documentsTableHtml` to year→category; add `trends` filter + Chart.js render (modify).
- `package.json` — add `chart.js` dependency (modify).

**Tests**
- `tests/test_documents_service.py`, `tests/test_api_mapping.py`, `tests/test_api_browse.py` (extend).
- `tests/test_segment_service.py` — **new** (or extend existing segment tests if present).
- `tests/test_trends_service.py` — **new**.

---

## Conventions

- Run backend tests with `TEST_DATABASE_URL` set (conftest refuses otherwise):
  `pytest tests/<file>::<test> -v`
- Frontend type-check: `cd medagentic-dashboard && npm run lint`
- All dynamic HTML in `main.ts` goes through the existing `esc()` escaper and the
  `setHtml(el, html)` helper — never assign markup to a raw element property.
- Commit after each task's tests are green.

---

### Task 1: Persist `classification` on documents and expose it as `category`

**Files:**
- Modify: `app/services/documents.py`
- Modify: `app/services/browse.py:82-93` (`list_documents_timeline`)
- Modify: `app/api/mapping.py:93-106` (`document_to_out`)
- Modify: `app/api/schemas.py` (`DocumentOut`)
- Test: `tests/test_documents_service.py`, `tests/test_api_mapping.py`

- [ ] **Step 1: Write failing test for `create_document` classification**

Add to `tests/test_documents_service.py`:

```python
def test_create_document_persists_classification():
    from app.db import SessionLocal
    from app.models import Patient
    from app.services.documents import create_document
    db = SessionLocal()
    try:
        p = Patient(name="Classify Person")
        db.add(p); db.flush()
        doc = create_document(db, patient_id=p.id, doc_type="lab report",
                              classification="Hematology")
        assert doc.classification == "Hematology"
    finally:
        db.close()
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest tests/test_documents_service.py::test_create_document_persists_classification -v`
Expected: FAIL — `create_document() got an unexpected keyword argument 'classification'`

- [ ] **Step 3: Add the param**

In `app/services/documents.py`, extend the signature and constructor:

```python
def create_document(db: Session, *, patient_id: int, doc_type: str | None = None,
                    source_type: str | None = None, mime_type: str | None = None,
                    file_path: str | None = None, content_hash: str | None = None,
                    report_date: dt.date | None = None,
                    original_name: str | None = None,
                    classification: str | None = None) -> Document:
    doc = Document(
        patient_id=patient_id,
        doc_type=doc_type,
        classification=classification,
        source_type=source_type,
        mime_type=mime_type,
        file_path=file_path,
        content_hash=content_hash,
        report_date=report_date,
        original_name=original_name,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc
```

- [ ] **Step 4: Run it, verify it passes**

Run: `pytest tests/test_documents_service.py::test_create_document_persists_classification -v`
Expected: PASS

- [ ] **Step 5: Write failing test for `document_to_out` category**

Add to `tests/test_api_mapping.py`:

```python
def test_document_to_out_includes_category():
    from app.api.mapping import document_to_out
    row = {"id": 7, "original_name": "cbc.pdf", "type": "lab report",
           "report_date": "2025-03-04", "date": None, "classification": "Hematology"}
    out = document_to_out(row, "0.1 MB")
    assert out["category"] == "Hematology"

def test_document_to_out_category_none_when_missing():
    from app.api.mapping import document_to_out
    out = document_to_out({"id": 8, "type": "FILE"}, "—")
    assert out["category"] is None
```

- [ ] **Step 6: Run them, verify they fail**

Run: `pytest tests/test_api_mapping.py -k category -v`
Expected: FAIL — `KeyError: 'category'`

- [ ] **Step 7: Emit category from `document_to_out`**

In `app/api/mapping.py`, add to the returned dict in `document_to_out`:

```python
    return {
        "id": str(row.get("id")),
        "name": name,
        "date": date,
        "type": (row.get("type") or "FILE"),
        "size": size_str,
        "category": row.get("classification"),
    }
```

- [ ] **Step 8: Surface `classification` from the timeline query**

In `app/services/browse.py`, in `list_documents_timeline`'s returned dict, add `"classification": d.classification`:

```python
        {"id": d.id, "patient": pname, "type": d.doc_type, "status": d.status,
         "classification": d.classification,
         "report_date": d.report_date.strftime("%Y-%m-%d") if d.report_date else None,
         "date": d.uploaded_at.strftime("%Y-%m-%d %H:%M") if d.uploaded_at else None,
         "original_name": d.original_name, "file": d.file_path}
```

- [ ] **Step 9: Add `category` to the `DocumentOut` schema**

In `app/api/schemas.py`:

```python
class DocumentOut(BaseModel):
    id: str
    name: str
    date: str | None = None
    type: str
    size: str
    category: str | None = None
```

- [ ] **Step 10: Run mapping tests, verify pass**

Run: `pytest tests/test_api_mapping.py -k category -v`
Expected: PASS

- [ ] **Step 11: Commit**

```bash
git add app/services/documents.py app/services/browse.py app/api/mapping.py app/api/schemas.py tests/test_documents_service.py tests/test_api_mapping.py
git commit -m "feat(docs): persist + expose document classification as category"
```

---

### Task 2: LLM-extract report category during segmentation

**Files:**
- Modify: `app/services/segment.py` (`ReportSpec`, `_SPLIT_PROMPT`, `split_reports`, `_regex_segments`)
- Modify: `app/agent/nodes/ingest.py` (pass category into `create_document`)
- Test: `tests/test_segment_service.py` (new)

- [ ] **Step 1: Write failing test for category passthrough**

Create `tests/test_segment_service.py`:

```python
from app.services.segment import split_reports, ReportSpec, _ReportSplit


class _FakeChat:
    """Returns a fixed structured split, ignoring the prompt."""
    def __init__(self, split): self._split = split
    def structured(self, prompt, schema): return self._split


def test_split_reports_carries_category():
    split = _ReportSplit(reports=[
        ReportSpec(title="CBC", doc_type="lab report", category="Hematology", pages=[0]),
        ReportSpec(title="Chest X-Ray", doc_type="imaging", category="X-Ray", pages=[1]),
    ])
    out = split_reports(_FakeChat(split), ["cbc page", "xray page"])
    cats = [s["category"] for s in out]
    assert cats == ["Hematology", "X-Ray"]


def test_regex_fallback_category_none():
    # single page -> no LLM call, category must be present and None
    out = split_reports(_FakeChat(_ReportSplit(reports=[])), ["only one page"])
    assert out[0]["category"] is None
```

- [ ] **Step 2: Run it, verify it fails**

Run: `pytest tests/test_segment_service.py -v`
Expected: FAIL — `ReportSpec` has no field `category` / `KeyError: 'category'`

- [ ] **Step 3: Add `category` to the schema + prompt**

In `app/services/segment.py`, extend `ReportSpec`:

```python
class ReportSpec(BaseModel):
    title: str = "Medical Report"      # human name, e.g. "Haematological Report", "Chest X-Ray", "Prescription"
    doc_type: str = "document"         # category: lab report | imaging | prescription | discharge | document
    category: str | None = None        # printed department/panel, e.g. Hematology, Biochemistry, X-Ray, Urine
    pages: list[int] = Field(default_factory=list)  # 0-based page indices in this report
    date: str | None = None            # report/collection date if visible
```

Update `_SPLIT_PROMPT` — add a `category` bullet to the per-report list, right after the `doc_type` bullet:

```python
- category: the report's printed department/section/panel name if shown (e.g. "Hematology", "Biochemistry", "X-Ray", "Ultrasound", "Urine"), else null
```

- [ ] **Step 4: Emit `category` from both code paths in `split_reports`**

In `_regex_segments`, add `"category": None` to the appended dict:

```python
            segs.append({"title": title, "doc_type": doc_type_for(title),
                         "category": None,
                         "date": None, "text": page, "pages": [i]})
```

In the single-page early return:

```python
        return [{"title": None, "doc_type": "document", "category": None,
                 "date": None, "text": text, "pages": [0]}]
```

In the LLM `out` builder loop:

```python
        out.append({
            "title": (r.title or "").strip() or None,
            "doc_type": (r.doc_type or "document").strip().lower(),
            "category": (r.category or "").strip() or None,
            "date": r.date,
            "text": "\n\n".join(pages[i] for i in idxs),
            "pages": idxs,
        })
```

- [ ] **Step 5: Run segment tests, verify pass**

Run: `pytest tests/test_segment_service.py -v`
Expected: PASS

- [ ] **Step 6: Pass category into document creation**

In `app/agent/nodes/ingest.py`, find the `create_document(` call (around line 282) and add `classification=seg.get("category")`:

```python
                doc = create_document(
                    s, patient_id=pid, doc_type=seg.get("doc_type") or ex.get("doc_type"),
                    classification=seg.get("category"),
                    ...  # keep all existing args unchanged
                )
```

Also, where the per-report segment dict is assembled earlier in ingest (the loop over `split_reports`, around line 128-145), carry the field through so `seg["category"]` survives to creation. Add to that dict literal, mirroring the existing `"doc_type": ...` line:

```python
            "category": seg.get("category"),
```

- [ ] **Step 7: Type/smoke check ingest imports**

Run: `python -c "import app.agent.nodes.ingest"`
Expected: no error (module imports cleanly).

- [ ] **Step 8: Commit**

```bash
git add app/services/segment.py app/agent/nodes/ingest.py tests/test_segment_service.py
git commit -m "feat(segment): LLM-extract report category into classification"
```

---

### Task 3: Trend-series service (`app/services/trends.py`)

**Files:**
- Create: `app/services/trends.py`
- Test: `tests/test_trends_service.py`

This is the shared data source. Pure functions over a DB session; reuses
`browse.list_test_results` for raw rows.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_trends_service.py`:

```python
import datetime as dt

from app.db import SessionLocal
from app.models import (
    Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services import trends


def _add_result(db, patient, doc_date, test_name, value, unit, ref):
    doc = Document(patient_id=patient.id, doc_type="lab report",
                   report_date=doc_date)
    db.add(doc); db.flush()
    mt = MedicalTest(name=test_name)
    db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value=value, unit=unit,
                    reference_range=ref)
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                          entity_id=tr.id))
    db.commit()


def test_list_metrics_needs_two_numeric_points():
    db = SessionLocal()
    try:
        p = Patient(name="Trend One"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2024, 1, 1), "Hemoglobin", "12", "g/dL", "12-16")
        _add_result(db, p, dt.date(2025, 1, 1), "Hemoglobin", "11", "g/dL", "12-16")
        _add_result(db, p, dt.date(2025, 1, 1), "Glucose", "90", "mg/dL", "70-100")  # only 1 point
        metrics = trends.list_metrics(db, p.id)
        keys = {m["key"] for m in metrics}
        assert "hemoglobin" in keys
        assert "glucose" not in keys
        hgb = next(m for m in metrics if m["key"] == "hemoglobin")
        assert hgb["label"] == "Hemoglobin"
        assert hgb["unit"] == "g/dL"
        assert hgb["n"] == 2
    finally:
        db.close()


def test_metric_series_sorted_with_range_flags():
    db = SessionLocal()
    try:
        p = Patient(name="Trend Two"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2025, 1, 1), "Hemoglobin", "11", "g/dL", "12-16")  # below
        _add_result(db, p, dt.date(2024, 1, 1), "Hemoglobin", "14", "g/dL", "12-16")  # in
        s = trends.metric_series(db, p.id, "hemoglobin")
        assert s["ref_low"] == 12.0 and s["ref_high"] == 16.0
        # sorted ascending by date
        assert [pt["date"] for pt in s["points"]] == ["2024-01-01", "2025-01-01"]
        assert [pt["value"] for pt in s["points"]] == [14.0, 11.0]
        assert [pt["in_range"] for pt in s["points"]] == [True, False]
    finally:
        db.close()


def test_metric_series_skips_non_numeric_and_undated():
    db = SessionLocal()
    try:
        p = Patient(name="Trend Three"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2025, 1, 1), "Culture", "Positive", "", "")
        _add_result(db, p, None, "Culture", "5", "", "")  # undated -> excluded
        s = trends.metric_series(db, p.id, "culture")
        assert s["points"] == []
        assert s["ref_low"] is None and s["ref_high"] is None
    finally:
        db.close()
```

- [ ] **Step 2: Run them, verify they fail**

Run: `pytest tests/test_trends_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.trends'`

- [ ] **Step 3: Implement the service**

Create `app/services/trends.py`:

```python
"""Trend series for numeric test results — the single data source of truth for
the Trends tab (Chart.js now) and the future PDF report (matplotlib). Pure
functions over a DB session; no HTTP, no rendering.

A "metric" is a test name normalized to lower/trimmed. Only metrics with >=2
numeric, dated data points are exposed (a single point is not a trend).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from sqlalchemy.orm import Session

from app.services import browse as bsvc

# leading signed float, e.g. "11.2", "-0.5", ".01"
_NUM_RE = re.compile(r"-?\d*\.?\d+")
# "low-high" interval, e.g. "12-16", "12.0 – 16.0"
_REF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")


def _num(value: str | None) -> float | None:
    if not value:
        return None
    m = _NUM_RE.match(value.strip())
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _ref(reference: str | None) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    m = _REF_RE.search(reference)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _key(name: str) -> str:
    return (name or "").strip().lower()


def list_metrics(db: Session, patient_id: int) -> list[dict]:
    """[{key, label, unit, n}] for tests with >=2 numeric, dated points."""
    rows = bsvc.list_test_results(db, patient_id=patient_id)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if _num(r.get("value")) is None or not r.get("date"):
            continue
        by_key[_key(r.get("test") or "")].append(r)
    metrics = []
    for key, items in by_key.items():
        if not key or len(items) < 2:
            continue
        label = Counter(r["test"] for r in items if r.get("test")).most_common(1)[0][0]
        unit = next((r["unit"] for r in items if r.get("unit")), "")
        metrics.append({"key": key, "label": label, "unit": unit, "n": len(items)})
    metrics.sort(key=lambda m: m["label"].lower())
    return metrics


def metric_series(db: Session, patient_id: int, key: str) -> dict:
    """{key,label,unit,ref_low,ref_high,points:[{date,value,in_range}]} for one
    metric. Points: numeric + dated only, sorted ascending by date."""
    rows = [r for r in bsvc.list_test_results(db, patient_id=patient_id)
            if _key(r.get("test") or "") == _key(key)]
    label = next((r["test"] for r in rows if r.get("test")), key)
    unit = next((r["unit"] for r in rows if r.get("unit")), "")
    ref_low = ref_high = None
    for r in rows:
        lo, hi = _ref(r.get("reference") or r.get("reference_range"))
        if lo is not None:
            ref_low, ref_high = lo, hi
            break
    points = []
    for r in rows:
        v = _num(r.get("value"))
        if v is None or not r.get("date"):
            continue
        in_range = True
        if ref_low is not None and ref_high is not None:
            in_range = ref_low <= v <= ref_high
        points.append({"date": r["date"], "value": v, "in_range": in_range})
    points.sort(key=lambda p: p["date"])
    return {"key": _key(key), "label": label, "unit": unit,
            "ref_low": ref_low, "ref_high": ref_high, "points": points}
```

- [ ] **Step 4: Run trends tests, verify pass**

Run: `pytest tests/test_trends_service.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/trends.py tests/test_trends_service.py
git commit -m "feat(trends): trend-series service for numeric test results"
```

---

### Task 4: Trend API routes + schemas

**Files:**
- Modify: `app/api/schemas.py` (add `MetricOut`, `SeriesPointOut`, `SeriesOut`)
- Modify: `app/api/routes_browse.py` (two routes)
- Test: `tests/test_api_browse.py`

- [ ] **Step 1: Write failing route tests**

Add to `tests/test_api_browse.py`:

```python
def test_trends_endpoints():
    import datetime as dt
    from app.db import SessionLocal
    from app.models import Document, DocumentEntity, MedicalTest, Patient, TestResult

    pid = int(client.post("/api/patients", json={"name": "Trend Api"}).json()["id"])
    db = SessionLocal()
    try:
        for d, val in ((dt.date(2024, 1, 1), "14"), (dt.date(2025, 1, 1), "11")):
            doc = Document(patient_id=pid, doc_type="lab report", report_date=d)
            db.add(doc); db.flush()
            mt = MedicalTest(name="Hemoglobin"); db.add(mt); db.flush()
            tr = TestResult(medical_test_id=mt.id, value=val, unit="g/dL",
                            reference_range="12-16")
            db.add(tr); db.flush()
            db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                                  entity_id=tr.id))
        db.commit()
    finally:
        db.close()

    metrics = client.get(f"/api/patients/{pid}/trends").json()
    assert any(m["key"] == "hemoglobin" and m["n"] == 2 for m in metrics)

    series = client.get(f"/api/patients/{pid}/trends/hemoglobin").json()
    assert series["ref_low"] == 12.0
    assert len(series["points"]) == 2
    assert series["points"][0]["date"] == "2024-01-01"


def test_trends_unknown_patient_404():
    assert client.get("/api/patients/999999/trends").status_code == 404
```

- [ ] **Step 2: Run them, verify they fail**

Run: `pytest tests/test_api_browse.py -k trends -v`
Expected: FAIL — 404/route not found (no such endpoint yet)

- [ ] **Step 3: Add the schemas**

In `app/api/schemas.py`:

```python
class MetricOut(BaseModel):
    key: str
    label: str
    unit: str = ""
    n: int


class SeriesPointOut(BaseModel):
    date: str
    value: float
    in_range: bool


class SeriesOut(BaseModel):
    key: str
    label: str
    unit: str = ""
    ref_low: float | None = None
    ref_high: float | None = None
    points: list[SeriesPointOut]
```

- [ ] **Step 4: Add the routes**

In `app/api/routes_browse.py`, add the service import and merge the new schema
names into the existing `from app.api.schemas import ...` line, then add these
routes after `patient_records`:

```python
from app.services import trends as tsvc
# merge MetricOut, SeriesOut into the existing schemas import line


@router.get("/patients/{patient_id}/trends", response_model=list[MetricOut])
def patient_trends(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return tsvc.list_metrics(db, patient_id)
    finally:
        db.close()


@router.get("/patients/{patient_id}/trends/{key}", response_model=SeriesOut)
def patient_trend_series(patient_id: int, key: str) -> dict:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return tsvc.metric_series(db, patient_id, key)
    finally:
        db.close()
```

(`psvc`, `SessionLocal`, `HTTPException`, `router` are already imported in this file.)

- [ ] **Step 5: Run route tests, verify pass**

Run: `pytest tests/test_api_browse.py -k trends -v`
Expected: PASS

- [ ] **Step 6: Full backend suite green**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add app/api/schemas.py app/api/routes_browse.py tests/test_api_browse.py
git commit -m "feat(api): trend metric + series endpoints"
```

---

### Task 5: Frontend types + API client

**Files:**
- Modify: `medagentic-dashboard/src/types.ts`
- Modify: `medagentic-dashboard/src/api.ts`

- [ ] **Step 1: Add `category` to `ApiDocument` and trend types**

In `src/types.ts`, add `category` to `ApiDocument`:

```typescript
export interface ApiDocument {
  id: string;
  name: string;
  date: string | null;
  type: string;
  size: string;
  category: string | null;
}
```

Append trend types:

```typescript
export interface TrendMetric {
  key: string;
  label: string;
  unit: string;
  n: number;
}

export interface TrendPoint {
  date: string;
  value: number;
  in_range: boolean;
}

export interface TrendSeries {
  key: string;
  label: string;
  unit: string;
  ref_low: number | null;
  ref_high: number | null;
  points: TrendPoint[];
}
```

- [ ] **Step 2: Add API client functions**

In `src/api.ts`, extend the import and add two fetchers:

```typescript
import {
  ApiDocument, ApiPatient, ApiRecord, Health, SseHandlers, TrendMetric, TrendSeries,
} from './types';

export const getTrendMetrics = (patientId: string) =>
  json<TrendMetric[]>(`/api/patients/${patientId}/trends`);
export const getTrendSeries = (patientId: string, key: string) =>
  json<TrendSeries>(`/api/patients/${patientId}/trends/${encodeURIComponent(key)}`);
```

- [ ] **Step 3: Type-check**

Run: `cd medagentic-dashboard && npm run lint`
Expected: no errors.

- [ ] **Step 4: Commit**

```bash
git add medagentic-dashboard/src/types.ts medagentic-dashboard/src/api.ts
git commit -m "feat(ui): trend types + api client, document category"
```

---

### Task 6: All tab → Year → category → documents

**Files:**
- Create: `medagentic-dashboard/src/grouping.ts`
- Modify: `medagentic-dashboard/src/main.ts` (`documentsTableHtml`, imports)

Isolate the bucket logic in a pure module so it is type-checkable and reusable.

- [ ] **Step 1: Create the pure grouping module**

Create `medagentic-dashboard/src/grouping.ts`:

```typescript
import { ApiDocument } from './types';

export interface CategoryGroup { category: string; docs: ApiDocument[]; }
export interface YearGroup { year: string; total: number; categories: CategoryGroup[]; }

// Year of report date (fallback upload date). Undated -> "Undated".
function yearOf(d: ApiDocument): string {
  if (!d.date) return 'Undated';
  const y = new Date(d.date).getFullYear();
  return Number.isNaN(y) ? 'Undated' : String(y);
}

function categoryOf(d: ApiDocument): string {
  return (d.category && d.category.trim())
    || (d.type && d.type.trim())
    || 'Uncategorized';
}

// Group documents into years, each year into categories. `desc` => newest year
// first. "Undated" always sorts last. Docs within a category keep input order.
export function groupDocsByYear(docs: ApiDocument[], desc: boolean): YearGroup[] {
  const years = new Map<string, Map<string, ApiDocument[]>>();
  for (const d of docs) {
    const y = yearOf(d);
    const cat = categoryOf(d);
    const byCat = years.get(y) ?? years.set(y, new Map()).get(y)!;
    (byCat.get(cat) ?? byCat.set(cat, []).get(cat)!).push(d);
  }
  const out: YearGroup[] = [...years.entries()].map(([year, byCat]) => {
    const categories: CategoryGroup[] = [...byCat.entries()]
      .map(([category, ds]) => ({ category, docs: ds }))
      .sort((a, b) => a.category.localeCompare(b.category));
    const total = categories.reduce((n, c) => n + c.docs.length, 0);
    return { year, total, categories };
  });
  out.sort((a, b) => {
    if (a.year === 'Undated') return 1;
    if (b.year === 'Undated') return -1;
    return desc ? b.year.localeCompare(a.year) : a.year.localeCompare(b.year);
  });
  return out;
}
```

- [ ] **Step 2: Rewrite `documentsTableHtml` to render year groups**

In `src/main.ts`, add the import near the top (with the other local imports):

```typescript
import { groupDocsByYear } from './grouping';
```

Replace the body of `documentsTableHtml` (the flat table at ~286-312) with
year → category → document rows. Reuse `expandedCards` + the `card-toggle`
class (keys prefixed `year:`) so the existing `bindCardButtons` toggle handler
drives expansion; reuse `dateColor`, `formatDate`, `docFileUrl`, `esc`,
`tableShell`. All markup is built as strings and returned through `tableShell`
(which is rendered via the existing `setHtml`), so dynamic values pass `esc()`:

```typescript
function documentsTableHtml(list: ApiDocument[]): string {
  const groups = groupDocsByYear(list, sortOrder === 'desc');
  const head = `
    <div class="grid grid-cols-[1fr_6rem] gap-3 items-center px-4 h-10 text-[10px] uppercase tracking-widest text-[#A6A298] font-bold">
      <span>Year</span><span class="text-right">Reports</span>
    </div>`;
  const rows = groups.map(g => {
    const key = `year:${g.year}`;
    const open = expandedCards.has(key);
    const body = open ? `
      <div class="px-4 pb-3 pt-1 bg-[#FCFBF8] border-b border-[#F4F3EF]">
        ${g.categories.map(c => `
          <div class="mt-2 first:mt-1">
            <div class="text-[10px] uppercase tracking-widest text-[#A6A298] font-bold px-1 pb-1">
              ${esc(c.category)} <span class="text-[#C9C6BD]">(${c.docs.length})</span>
            </div>
            ${c.docs.map(d => docRowHtml(d)).join('')}
          </div>`).join('')}
      </div>` : '';
    return `
      <div class="border-b border-[#F4F3EF] last:border-0">
        <button class="card-toggle w-full grid grid-cols-[1fr_6rem] gap-3 items-center px-4 min-h-[44px] py-1.5 text-left hover:bg-[#FAF9F5] transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F] focus-visible:ring-inset ${open ? 'bg-[#FAF9F5]' : ''}" data-key="${esc(key)}" aria-expanded="${open}">
          <span class="flex items-center gap-2 min-w-0">
            <i data-lucide="chevron-${open ? 'down' : 'right'}" class="w-4 h-4 text-[#A6A298] shrink-0"></i>
            <span class="text-[13px] font-semibold text-[#2E2C29]">${esc(g.year)}</span>
          </span>
          <span class="text-[12px] text-[#8C8982] tabular-nums text-right">${g.total}</span>
        </button>
        ${body}
      </div>`;
  }).join('');
  return tableShell(head, rows);
}

// A single document line inside an expanded year/category group.
function docRowHtml(d: ApiDocument): string {
  const color = d.date ? dateColor(d.date) : '#e6bb4d';
  const url = docFileUrl(d.id);
  return `
    <div class="group grid grid-cols-[1fr_5.5rem_4.5rem] sm:grid-cols-[1fr_7rem_5rem] gap-3 items-center px-2 min-h-[40px] py-1 hover:bg-[#FAF9F5] rounded-lg transition-colors duration-150">
      <a href="${esc(url)}" target="_blank" rel="noopener" title="${esc(d.name)}"
         class="flex items-center gap-2.5 min-w-0 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]">
        <span class="w-1.5 h-1.5 rounded-full shrink-0" style="background:${color}"></span>
        <i data-lucide="file-text" class="w-4 h-4 text-[#8C8982] shrink-0"></i>
        <span class="text-[13px] font-medium text-[#2E2C29] truncate">${esc(d.name)}</span>
      </a>
      <span class="text-[12px] text-[#59554D] tabular-nums text-right whitespace-nowrap">${d.date ? esc(formatDate(d.date)) : '—'}</span>
      <span class="flex items-center gap-0.5 justify-end">
        <a href="${esc(url)}" target="_blank" rel="noopener" aria-label="Open PDF" title="Open PDF"
           class="w-8 h-8 flex items-center justify-center text-[#5D7B6F] hover:text-[#3f5b50] rounded-lg hover:bg-[#EEF2F0]"><i data-lucide="external-link" class="w-4 h-4"></i></a>
        <button class="del-doc w-8 h-8 flex items-center justify-center text-[#C0857A] hover:text-[#a3553f] rounded-lg hover:bg-[#F5EDE9]" data-id="${esc(d.id)}" data-label="${esc(d.name)}" aria-label="Delete document" title="Delete"><i data-lucide="trash-2" class="w-4 h-4"></i></button>
      </span>
    </div>`;
}
```

(The `.del-doc` buttons keep the same `data-id`/`data-label`, so the existing
delete handler in `bindCardButtons` works unchanged. `card-toggle` with the
`year:` key reuses the existing expand handler.)

- [ ] **Step 3: Type-check**

Run: `cd medagentic-dashboard && npm run lint`
Expected: no errors.

- [ ] **Step 4: Manual verification**

Start backend + frontend (`make` targets or `npm run dev`). Select a patient
with documents. Confirm the All tab shows year rows; expanding a year reveals
category sub-headers and document rows; Open and Delete still work; the
Newest/Oldest toggle flips year order; undated documents appear under "Undated"
last.

- [ ] **Step 5: Commit**

```bash
git add medagentic-dashboard/src/grouping.ts medagentic-dashboard/src/main.ts
git commit -m "feat(ui): All tab grouped by year then category"
```

---

### Task 7: Trends tab (Chart.js)

**Files:**
- Modify: `medagentic-dashboard/package.json` (add chart.js)
- Modify: `medagentic-dashboard/src/main.ts` (filter bar, trends render, state)

- [ ] **Step 1: Install Chart.js**

Run:

```bash
cd medagentic-dashboard && npm install chart.js
```

Expected: `chart.js` added to `dependencies` in `package.json`.

- [ ] **Step 2: Add `trends` to the filter bar**

In `src/main.ts`, change the filters array (~196):

```typescript
    const filters = ['all', 'trends', 'disease', 'symptom', 'medicine', 'test_result'];
```

The existing label logic (`type === 'all' ? 'All' : type.replace('_', ' ')` with
a `capitalize` class) renders "trends" → "Trends" automatically. No label change
needed.

- [ ] **Step 3: Add module-level trend state + Chart import**

Near the top of `src/main.ts` add the import:

```typescript
import { Chart, registerables } from 'chart.js';
Chart.register(...registerables);
```

Add module-level state alongside the other `let` declarations (~36-42):

```typescript
let trendMetric: string | null = null;
let trendChart: Chart | null = null;
```

- [ ] **Step 4: Branch the dashboard render for `trends`**

In `renderDashboard`, handle trends before the `all` / entity branches (~227):

```typescript
  if (filterType === 'trends') {
    setHtml(grid, trendsShellHtml());
    bindTrendControls();
    return;
  }
  if (filterType === 'all') {
    // ...existing all branch unchanged...
```

- [ ] **Step 5: Add the Trends shell + control binding + chart render**

Add these functions to `src/main.ts`. The `<select>` options are built as an
escaped string and assigned through the existing `setHtml` helper (never a raw
property):

```typescript
function trendsShellHtml(): string {
  return `
    <div class="bg-white rounded-2xl border border-[#E0DDD5] shadow-sm p-5">
      <div class="flex items-center gap-3 mb-4">
        <span class="text-[10px] uppercase tracking-widest text-[#A6A298] font-bold">Metric</span>
        <select id="trend-metric" class="text-[13px] font-semibold text-[#2E2C29] bg-[#FAF9F5] border border-[#E0DDD5] rounded-lg px-3 py-1.5 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5D7B6F]"></select>
        <span id="trend-unit" class="text-[12px] text-[#8C8982]"></span>
      </div>
      <div class="relative h-[320px]"><canvas id="trend-canvas"></canvas></div>
      <p id="trend-empty" class="hidden text-center py-16 text-[#A6A298] text-sm"></p>
    </div>`;
}

async function bindTrendControls() {
  if (!currentPatientId) return;
  const sel = $('trend-metric') as HTMLSelectElement | null;
  const empty = $('trend-empty');
  const metrics = await getTrendMetrics(currentPatientId).catch(() => []);
  if (!metrics.length) {
    if (sel) sel.classList.add('hidden');
    const canvas = $('trend-canvas'); if (canvas) canvas.classList.add('hidden');
    if (empty) {
      empty.classList.remove('hidden');
      empty.textContent = 'No trend data yet — a test needs ≥2 numeric results to chart.';
    }
    return;
  }
  if (!trendMetric || !metrics.some(m => m.key === trendMetric)) {
    trendMetric = metrics[0].key;
  }
  if (sel) {
    setHtml(sel, metrics.map(m =>
      `<option value="${esc(m.key)}" ${m.key === trendMetric ? 'selected' : ''}>${esc(m.label)}</option>`).join(''));
    sel.addEventListener('change', () => { trendMetric = sel.value; renderTrendChart(); });
  }
  renderTrendChart();
}

async function renderTrendChart() {
  if (!currentPatientId || !trendMetric) return;
  const s = await getTrendSeries(currentPatientId, trendMetric).catch(() => null);
  const unitEl = $('trend-unit');
  if (unitEl) unitEl.textContent = s?.unit ? `(${s.unit})` : '';
  const canvas = $('trend-canvas') as HTMLCanvasElement | null;
  if (!canvas || !s) return;
  if (trendChart) { trendChart.destroy(); trendChart = null; }

  const labels = s.points.map(p => p.date);
  const values = s.points.map(p => p.value);
  const pointColors = s.points.map(p => (p.in_range ? '#5D7B6F' : '#C16D54'));
  const datasets: any[] = [{
    label: s.label, data: values, borderColor: '#5D7B6F',
    backgroundColor: '#5D7B6F', pointBackgroundColor: pointColors,
    pointRadius: 4, tension: 0.25, fill: false,
  }];
  // Shaded reference band: two flat hidden lines with fill between them.
  if (s.ref_low !== null && s.ref_high !== null) {
    datasets.push(
      { label: 'ref high', data: labels.map(() => s.ref_high), borderWidth: 0,
        pointRadius: 0, fill: '+1', backgroundColor: 'rgba(93,123,111,0.10)' },
      { label: 'ref low', data: labels.map(() => s.ref_low), borderWidth: 0,
        pointRadius: 0, fill: false },
    );
  }
  trendChart = new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: false } },
    },
  });
}
```

Add `getTrendMetrics, getTrendSeries` to the existing `import { ... } from './api'` line.

- [ ] **Step 6: Reset chart state on patient switch**

In the function that loads a patient's data (where `records`/`docs` are fetched
and `filterType` is reset to `'all'`, ~106-115), also reset trend state so a
stale chart never persists across patients:

```typescript
  trendMetric = null;
  if (trendChart) { trendChart.destroy(); trendChart = null; }
```

- [ ] **Step 7: Type-check**

Run: `cd medagentic-dashboard && npm run lint`
Expected: no errors.

- [ ] **Step 8: Manual verification**

Run backend + frontend. Pick a patient with ≥2 numeric results for some test.
Click **Trends**: dropdown lists metrics; chart draws a line; switching metrics
re-renders without a full reload; reference band is shaded; out-of-range points
are red; a patient with no qualifying metric shows the empty message; switching
patients does not leave a stale chart.

- [ ] **Step 9: Commit**

```bash
git add medagentic-dashboard/package.json medagentic-dashboard/package-lock.json medagentic-dashboard/src/main.ts
git commit -m "feat(ui): Trends tab with Chart.js line + reference band"
```

---

## Self-Review Notes

- **Spec coverage:** Unit 1 → Tasks 1–2; Unit 2 → Task 3; trend routes → Task 4;
  frontend plumbing → Task 5; Unit 3 (year/category All tab) → Task 6; Unit 4
  (Trends tab) → Task 7. Reference-band shading + red points → Task 7 Step 5.
  "Uncategorized" / "Undated" fallbacks → Task 6 grouping module.
- **Out of scope (per spec):** matplotlib renderer, PDF report, re-classify
  backfill, multi-metric overlay — not in any task, intentionally.
- **Type consistency:** `classification` (DB/service) → `category` (API/TS)
  mapping is deliberate and applied consistently. `groupDocsByYear(docs, desc)`,
  `getTrendMetrics`, `getTrendSeries`, `trendMetric`, `trendChart` names match
  across tasks. Series shape (`key/label/unit/ref_low/ref_high/points[]`) is
  identical in service (Task 3), schema (Task 4), and TS type (Task 5).
- **Frontend testing:** repo has no JS test runner; frontend tasks verify via
  `npm run lint` (tsc) + explicit manual steps. Bucket logic isolated in
  `grouping.ts` to keep it type-checkable and side-effect-free.
```
