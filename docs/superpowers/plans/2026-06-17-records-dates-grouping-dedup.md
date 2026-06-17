# Records: dates, dated tables, value cleanup, early dedup, delete — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Store each document's real report date, render records as compact per-date tables (color-coded), clean test value/reference fields, catch duplicate uploads by file hash before OCR, and allow deleting a date's records.

**Architecture:** Backend = Alembic migration + service/query/mapping changes + one new ingest graph node (`dedup_check`) before OCR + a purge service & delete route. Frontend = grouped tables in `main.ts` + a `deleteRecords` client call.

**Tech Stack:** SQLAlchemy, Alembic, LangGraph, FastAPI, pytest; Vite + vanilla TypeScript.

**Security note:** every dynamic DB/OCR/LLM string rendered into `innerHTML` MUST pass through the existing `esc()` helper in `main.ts`.

**DB note:** tests run against the live Supabase DB and `conftest._schema` uses `create_all` (no ALTER). So the migration in Task 1 must be **applied** (`alembic upgrade head`) before the persist/browse tests in later tasks.

---

## File Structure

**Backend:**
- `migrations/versions/0003_document_report_date.py` — new migration
- `app/models.py` — add `Document.report_date`
- `app/services/dates.py` — `parse_doc_date` (new)
- `app/services/entities.py` — persist `report_date` + `observed_at`
- `app/services/browse.py` — return/sort by `report_date`
- `app/services/purge.py` — `delete_documents` (new)
- `app/api/mapping.py` — `format_value`, unit/reference fields
- `app/api/schemas.py` — `RecordOut` gains `unit`, `reference`; `DeleteRecordsIn`
- `app/api/routes_browse.py` — `POST /patients/{id}/records/delete`
- `app/agent/nodes/ingest.py` — `dedup_check_node`
- `app/agent/graph.py` — wire `dedup_check`

**Frontend:**
- `medagentic-dashboard/src/types.ts` — `ApiRecord` gains `unit`, `reference`
- `medagentic-dashboard/src/api.ts` — `deleteRecords`
- `medagentic-dashboard/src/main.ts` — grouped tables, delete buttons

**Tests:** `tests/test_dates.py`, `tests/test_api_mapping.py` (append), `tests/test_entities_dates.py`, `tests/test_api_browse.py` (append), `tests/test_dedup.py`, `tests/test_purge.py`

---

## Task 1: Migration + model for report_date

**Files:**
- Create: `migrations/versions/0003_document_report_date.py`
- Modify: `app/models.py`

- [ ] **Step 1: Add the column to the model** — in `app/models.py`, inside `class Document`, after the `content_hash` line add:

```python
    report_date: Mapped["date | None"] = mapped_column(Date, nullable=True)  # date printed on the document
```

Add `Date` to the sqlalchemy import on line 4 (it currently imports `Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func`):

```python
from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, func
```

Add the `date` import at the top (line 1 currently `from datetime import datetime`):

```python
from datetime import date, datetime
```

- [ ] **Step 2: Create the migration**

```python
"""add document.report_date (date printed on the document)

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document", sa.Column("report_date", sa.Date(), nullable=True))


def downgrade() -> None:
    op.drop_column("document", "report_date")
```

- [ ] **Step 3: Apply it**

Run: `alembic upgrade head`
Expected: runs 0003, no error. (Adds `report_date` to the live DB so later tests see it.)

- [ ] **Step 4: Commit**

```bash
git add migrations/versions/0003_document_report_date.py app/models.py
git commit -m "feat(db): document.report_date + migration 0003"
```

---

## Task 2: Date parser

**Files:**
- Create: `app/services/dates.py`
- Test: `tests/test_dates.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_dates.py`:

```python
import datetime as dt

from app.services.dates import parse_doc_date


def test_iso():
    assert parse_doc_date("2023-10-05") == dt.date(2023, 10, 5)


def test_day_first_slash():
    assert parse_doc_date("05/10/2023") == dt.date(2023, 10, 5)


def test_textual():
    assert parse_doc_date("5 Oct 2023") == dt.date(2023, 10, 5)
    assert parse_doc_date("October 5, 2023") == dt.date(2023, 10, 5)


def test_two_digit_year_day_first():
    assert parse_doc_date("05-10-23") == dt.date(2023, 10, 5)


def test_unparseable_is_none():
    assert parse_doc_date("") is None
    assert parse_doc_date(None) is None
    assert parse_doc_date("not a date") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_dates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.dates'`

- [ ] **Step 3: Create `app/services/dates.py`**

```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_dates.py -v`
Expected: PASS (all date tests green)

- [ ] **Step 5: Commit**

```bash
git add app/services/dates.py tests/test_dates.py
git commit -m "feat(services): tolerant document date parser (day-first)"
```

---

## Task 3: Value formatter (mapping helper)

**Files:**
- Modify: `app/api/mapping.py`
- Test: `tests/test_api_mapping.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api_mapping.py`:

```python
def test_format_value_strips_glued_reference():
    val, ref = mapping.format_value("52  0-15", "mm/1hr", "")
    assert val == "52"
    assert ref == "0-15"


def test_format_value_keeps_existing_reference():
    val, ref = mapping.format_value("6.8 (4.0-6.0)", "%", "4.0-6.0")
    assert val == "6.8"
    assert ref == "4.0-6.0"  # provided ref wins; not overwritten


def test_format_value_normalizes_number():
    val, _ = mapping.format_value(".01", "", "")
    assert val == "0.01"


def test_format_value_passthrough_text():
    val, ref = mapping.format_value("Negative", "", "")
    assert val == "Negative"
    assert ref == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_mapping.py -v`
Expected: FAIL — `AttributeError: module 'app.api.mapping' has no attribute 'format_value'`

- [ ] **Step 3: Add `format_value` to `app/api/mapping.py`** — insert after the imports (after the `_ENTITY_TO_UI` line):

```python
import re

_REF_RE = re.compile(r"\(?\s*(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)\s*\)?")


def format_value(value: str | None, unit: str | None, reference: str | None) -> tuple[str, str]:
    """Return (clean_value, reference). Split a glued 'value  low-high' interval off the
    value; normalize numbers; pass non-numeric values through unchanged."""
    ref = (reference or "").strip()
    v = (value or "").strip()
    if not v:
        return "", ref
    # If a low-high interval is embedded in the value, peel it off.
    m = _REF_RE.search(v)
    if m and re.match(r"^\s*\d", v):
        interval = m.group(1).replace(" ", "")
        head = v[:m.start()].strip()
        if head:
            v = head
            if not ref:
                ref = interval
    # Normalize a leading bare number (".01" -> "0.01", "6.80" -> "6.8").
    nm = re.match(r"^-?\d*\.?\d+", v)
    if nm and nm.group(0) == v:
        try:
            f = float(v)
            v = ("%g" % f)
        except ValueError:
            pass
    return v, ref
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_mapping.py -v`
Expected: PASS (existing + new format_value tests green)

- [ ] **Step 5: Commit**

```bash
git add app/api/mapping.py tests/test_api_mapping.py
git commit -m "feat(api): format_value splits glued reference + normalizes numbers"
```

---

## Task 4: Persist report_date + observed_at

**Files:**
- Modify: `app/services/entities.py`
- Test: `tests/test_entities_dates.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_entities_dates.py`:

```python
import datetime as dt

from app.agent.state import ExtractedTest, ExtractionResult
from app.models import Document, Patient, TestResult
from app.services.entities import persist_extraction


def _doc(db):
    p = Patient(name="DateTest Person")
    db.add(p); db.flush()
    d = Document(patient_id=p.id, doc_type="lab")
    db.add(d); db.commit(); db.refresh(d)
    return d


def test_persist_sets_report_date_and_observed_at(db):
    d = _doc(db)
    res = ExtractionResult(doc_date="05/10/2023",
                           tests=[ExtractedTest(name="HbA1c", value="6.8", unit="%")])
    persist_extraction(db, document_id=d.id, result=res)
    db.refresh(d)
    assert d.report_date == dt.date(2023, 10, 5)
    tr = db.query(TestResult).order_by(TestResult.id.desc()).first()
    assert tr.observed_at is not None
    assert tr.observed_at.date() == dt.date(2023, 10, 5)


def test_persist_no_date_leaves_null(db):
    d = _doc(db)
    res = ExtractionResult(tests=[ExtractedTest(name="WBC", value="5")])
    persist_extraction(db, document_id=d.id, result=res)
    db.refresh(d)
    assert d.report_date is None
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_entities_dates.py -v`
Expected: FAIL — `report_date` stays None / `observed_at` None (assertion error).

- [ ] **Step 3: Update `app/services/entities.py`** — add imports and set the dates.

At the top, add:

```python
import datetime as dt

from app.models import Document
from app.services.dates import parse_doc_date
```

(Extend the existing `from app.models import (...)` line to include `Document`, or add the separate import above.)

In `persist_extraction`, immediately after `count = 0`, add:

```python
    observed = parse_doc_date(result.doc_date)
    if observed is not None:
        doc = db.get(Document, document_id)
        if doc is not None:
            doc.report_date = observed
    observed_dt = dt.datetime.combine(observed, dt.time()) if observed else None
```

Change the `TestResult(...)` construction (currently `value=t.value, unit=t.unit, reference_range=t.reference_range`) to also set the timestamp:

```python
        tr = TestResult(medical_test_id=mt.id, value=t.value, unit=t.unit,
                        reference_range=t.reference_range, observed_at=observed_dt)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_entities_dates.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/entities.py tests/test_entities_dates.py
git commit -m "feat(services): persist report_date + test observed_at from doc_date"
```

---

## Task 5: Browse returns + sorts by report_date

**Files:**
- Modify: `app/services/browse.py`
- Test: `tests/test_api_browse.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_api_browse.py`:

```python
import datetime as dt

from app.db import SessionLocal
from app.models import (
    Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services import browse as bsvc


def test_list_test_results_prefers_report_date():
    db = SessionLocal()
    try:
        p = Patient(name="Browse Date Person")
        db.add(p); db.flush()
        doc = Document(patient_id=p.id, doc_type="lab",
                       report_date=dt.date(2023, 10, 5))
        db.add(doc); db.flush()
        mt = MedicalTest(name="ESR")
        db.add(mt); db.flush()
        tr = TestResult(medical_test_id=mt.id, value="52", unit="mm/1hr",
                        reference_range="0-15")
        db.add(tr); db.flush()
        db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                              entity_id=tr.id, source_span="ESR 52"))
        db.commit()
        rows = bsvc.list_test_results(db, patient_id=p.id)
        assert rows[0]["date"] == "2023-10-05"  # report_date, not today
        assert rows[0]["reference_range"] == "0-15"
    finally:
        db.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_browse.py::test_list_test_results_prefers_report_date -v`
Expected: FAIL — date equals today's upload date, not `2023-10-05`.

- [ ] **Step 3: Update `app/services/browse.py`** — use `report_date` with an `uploaded_at` fallback in both query functions.

In `list_entity_links`: add `Document.report_date` to the selected columns (after `Document.uploaded_at`), change ordering to prefer it, and compute the date with a fallback. Replace the `.order_by(...)` line and the return comprehension:

```python
    q = q.order_by(func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(), model.name)
    return [
        {"name": r[0], "patient": r[1], "patient_id": r[2], "doc_type": r[3],
         "date": (r[8].strftime("%Y-%m-%d") if r[8]
                  else (r[4].strftime("%Y-%m-%d") if r[4] else None)),
         "confidence": round(r[5], 2) if r[5] is not None else None,
         "source": r[6], "document_id": r[7]}
        for r in q.all()
    ]
```

and add `Document.report_date,` to the `db.query(...)` column list (it becomes index `r[8]`). Add `from sqlalchemy import and_, func` (extend the existing `from sqlalchemy import and_`).

In `list_test_results`: likewise add `Document.report_date,` to the column list (after `Document.uploaded_at`) — it becomes the LAST column, index `r[10]`. Replace the order_by + return:

```python
    q = q.order_by(func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(), MedicalTest.name)
    return [
        {"test": r[0], "value": r[1], "unit": r[2], "reference_range": r[3],
         "patient": r[4], "patient_id": r[5], "doc_type": r[6],
         "date": (r[10].strftime("%Y-%m-%d") if r[10]
                  else (r[7].strftime("%Y-%m-%d") if r[7] else None)),
         "source": r[8], "document_id": r[9]}
        for r in q.all()
    ]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_browse.py -v`
Expected: PASS (existing browse tests + the new report_date test green)

- [ ] **Step 5: Commit**

```bash
git add app/services/browse.py tests/test_api_browse.py
git commit -m "feat(services): browse prefers report_date over upload date"
```

---

## Task 6: Mapping emits unit/reference + schema fields

**Files:**
- Modify: `app/api/schemas.py`
- Modify: `app/api/mapping.py`
- Test: `tests/test_api_mapping.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_api_mapping.py`:

```python
def test_merge_records_emits_unit_and_reference():
    tests = [{"test": "ESR", "value": "52  0-15", "unit": "mm/1hr",
              "reference_range": "", "source": "ESR 52", "doc_type": "lab",
              "date": "2023-10-05", "document_id": 3}]
    rows = mapping.merge_records("1", [], [], [], tests)
    tr = rows[0]
    assert tr["title"] == "ESR"
    assert tr["value"] == "52"
    assert tr["unit"] == "mm/1hr"
    assert tr["reference"] == "0-15"      # peeled off the glued value
    assert tr["date"] == "2023-10-05"


def test_merge_records_reference_range_wins():
    tests = [{"test": "HbA1c", "value": "6.8", "unit": "%",
              "reference_range": "4.0-6.0", "document_id": 4}]
    tr = mapping.merge_records("1", [], [], [], tests)[0]
    assert tr["value"] == "6.8"
    assert tr["reference"] == "4.0-6.0"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_mapping.py -v`
Expected: FAIL — `KeyError: 'unit'` / `'reference'` not in the record dict.

- [ ] **Step 3: Update `app/api/schemas.py`** — add two optional fields to `RecordOut` (after `title`/`description`):

```python
class RecordOut(BaseModel):
    id: str
    patientId: str
    type: str  # disease | symptom | medicine | test_result | treatment_plan
    title: str
    description: str
    value: str = ""
    unit: str = ""
    reference: str = ""
    date: str | None = None
    status: str = "Recorded"
    severity: str | None = None
    doctor: str | None = None
```

Add a delete-request model at the end of the file:

```python
class DeleteRecordsIn(BaseModel):
    document_ids: list[str]
```

- [ ] **Step 4: Update `app/api/mapping.py`** — give `_record` the new fields and use `format_value` for tests.

Change `_record` to accept value/unit/reference and include them (default empty):

```python
def _record(patient_id: str, ui_type: str, idx: int, title: str, row: dict,
            value: str = "", unit: str = "", reference: str = "") -> dict:
    return {
        "id": f"{ui_type}-{row.get('document_id')}-{idx}",
        "patientId": patient_id,
        "type": ui_type,
        "title": title,
        "description": (row.get("source") or row.get("doc_type") or ""),
        "value": value,
        "unit": unit,
        "reference": reference,
        "date": row.get("date"),
        "status": "Recorded",
        "severity": None,
        "doctor": None,
    }
```

Replace the test loop in `merge_records` (currently builds `value`/`title` and calls `_record`) with:

```python
    for r in tests:
        value, reference = format_value(r.get("value"), r.get("unit"), r.get("reference_range"))
        out.append(_record(patient_id, "test_result", idx, r.get("test") or "", r,
                           value=value, unit=(r.get("unit") or ""), reference=reference))
        idx += 1
    return out
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_api_mapping.py -v`
Expected: PASS (all mapping tests green). NOTE: the older `test_merge_records_maps_types_and_titles` asserts `tr["title"] == "HbA1c: 6.8%"`; update that one assertion to the new shape: `assert tr["title"] == "HbA1c"` and `assert tr["value"] == "6.8"` and `assert tr["unit"] == "%"`.

- [ ] **Step 6: Commit**

```bash
git add app/api/schemas.py app/api/mapping.py tests/test_api_mapping.py
git commit -m "feat(api): records carry clean value/unit/reference fields"
```

---

## Task 7: Early dedup node before OCR

**Files:**
- Modify: `app/agent/nodes/ingest.py`
- Modify: `app/agent/graph.py`
- Test: `tests/test_dedup.py`

The router currently routes `ingest -> extract_text`. Insert `dedup_check` so a byte-identical re-upload is caught from the file hash before OCR/LLM/HITL.

- [ ] **Step 1: Write the failing test** — create `tests/test_dedup.py`:

```python
import hashlib

from app.agent.nodes.ingest import dedup_check_node


class _FakeSession:
    def __init__(self, existing): self._existing = existing
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Deps:
    def __init__(self, existing): self._existing = existing
    def session_factory(self):
        return _FakeSession(self._existing)


def _write(tmp_path, data):
    f = tmp_path / "scan.png"
    f.write_bytes(data)
    return str(f)


def test_dedup_hit_short_circuits(tmp_path, monkeypatch):
    import app.agent.nodes.ingest as ing

    class Doc:  # stand-in for an existing Document row
        id = 11
        patient_id = 7

    monkeypatch.setattr(ing, "find_by_content_hash", lambda s, h: Doc())
    path = _write(tmp_path, b"PNGDATA")
    state = {"file_path": path, "messages": [{"role": "user", "content": "x"}]}
    out = dedup_check_node(state, {"configurable": {"deps": _Deps(Doc())}})
    assert out["already_ingested"] is True
    assert out["document_id"] == 11
    assert out["patient_id"] == 7
    assert out["dedup"] == "duplicate"


def test_dedup_miss_passes_hash_through(tmp_path, monkeypatch):
    import app.agent.nodes.ingest as ing
    monkeypatch.setattr(ing, "find_by_content_hash", lambda s, h: None)
    data = b"FRESHDATA"
    path = _write(tmp_path, data)
    out = dedup_check_node({"file_path": path, "messages": []},
                           {"configurable": {"deps": _Deps(None)}})
    assert out["dedup"] == "new"
    assert out["content_hash"] == hashlib.sha256(data).hexdigest()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_dedup.py -v`
Expected: FAIL — `ImportError: cannot import name 'dedup_check_node'`

- [ ] **Step 3: Add `dedup_check_node` to `app/agent/nodes/ingest.py`** — insert after `extract_text_node`:

```python
def dedup_check_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Hash the staged file BEFORE OCR. If this exact file was ingested already,
    short-circuit: reuse the existing document and skip OCR/extraction/HITL."""
    deps = config["configurable"]["deps"]
    data = Path(state["file_path"]).read_bytes()
    chash = hashlib.sha256(data).hexdigest()
    with deps.session_factory() as s:
        existing = find_by_content_hash(s, chash)
        dup = (existing.id, existing.patient_id) if existing is not None else None
    if dup is not None:
        staged = state.get("file_path")
        if staged and os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass
        return {"dedup": "duplicate", "already_ingested": True,
                "content_hash": chash, "document_id": dup[0], "patient_id": dup[1],
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": "This document was already on file — skipped to avoid duplicates.",
                }]}
    return {"dedup": "new", "content_hash": chash}
```

- [ ] **Step 4: Wire it into `app/agent/graph.py`** — register the node and reroute the ingest branch.

Add the import (extend the existing ingest-nodes import) for `dedup_check_node`, then add the node near the other ingest nodes:

```python
    g.add_node("dedup_check", dedup_check_node)
```

Change the router branch so ingest enters `dedup_check` first — in the `g.add_conditional_edges("router", _route, {...})` map, change `"ingest": "extract_text"` to `"ingest": "dedup_check"`.

Add a conditional edge out of `dedup_check` (insert before the `g.add_edge("extract_text", "extract_entities")` line):

```python
    g.add_conditional_edges("dedup_check", lambda s: s.get("dedup", "new"),
                            {"duplicate": END, "new": "extract_text"})
```

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_dedup.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Guard the build + commit**

Run: `python -c "from app.agent.graph import build_graph; build_graph(); print('graph ok')"`
Expected: prints `graph ok` (no edge/compile error).

```bash
git add app/agent/nodes/ingest.py app/agent/graph.py tests/test_dedup.py
git commit -m "feat(agent): hash-based dedup before OCR (skips extract+HITL on re-upload)"
```

---

## Task 8: Purge service (delete documents)

**Files:**
- Create: `app/services/purge.py`
- Test: `tests/test_purge.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_purge.py`:

```python
from app.db import SessionLocal
from app.models import (
    Chunk, Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services.purge import delete_documents


def _seed(db):
    p = Patient(name="Purge Person")
    db.add(p); db.flush()
    doc = Document(patient_id=p.id, doc_type="lab")
    db.add(doc); db.flush()
    mt = MedicalTest(name="WBC")
    db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value="5")
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                          entity_id=tr.id))
    db.add(Chunk(document_id=doc.id, patient_id=p.id, ord=0, text="hi"))
    db.commit()
    return p.id, doc.id, tr.id


def test_delete_documents_removes_doc_entities_chunks_testresults():
    db = SessionLocal()
    try:
        pid, doc_id, tr_id = _seed(db)
        n = delete_documents(db, pid, [str(doc_id)])
        assert n == 1
        assert db.get(Document, doc_id) is None
        assert db.get(TestResult, tr_id) is None
        assert db.query(DocumentEntity).filter_by(document_id=doc_id).count() == 0
        assert db.query(Chunk).filter_by(document_id=doc_id).count() == 0
    finally:
        db.close()


def test_delete_documents_skips_foreign_patient():
    db = SessionLocal()
    try:
        pid, doc_id, _ = _seed(db)
        other = Patient(name="Other")
        db.add(other); db.flush()
        db.commit()
        n = delete_documents(db, other.id, [str(doc_id)])  # wrong owner
        assert n == 0
        assert db.get(Document, doc_id) is not None
    finally:
        db.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_purge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.purge'`

- [ ] **Step 3: Create `app/services/purge.py`**

```python
from __future__ import annotations

import os

from sqlalchemy.orm import Session

from app.models import Document, DocumentEntity, TestResult


def delete_documents(db: Session, patient_id: int, document_ids: list[str]) -> int:
    """Delete the given documents (only those owned by patient_id) plus the
    TestResult rows they reference. DocumentEntity and Chunk cascade via FK.
    Shared name tables (Disease/Symptom/Medication/MedicalTest) are left intact.
    Returns the number of documents deleted."""
    ids = []
    for raw in document_ids:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not ids:
        return 0
    docs = (db.query(Document)
            .filter(Document.id.in_(ids), Document.patient_id == patient_id)
            .all())
    deleted = 0
    for doc in docs:
        # TestResult has no FK to Document — delete via the test_result links first.
        links = (db.query(DocumentEntity)
                 .filter(DocumentEntity.document_id == doc.id,
                         DocumentEntity.entity_type == "test_result")
                 .all())
        tr_ids = [l.entity_id for l in links]
        if tr_ids:
            (db.query(TestResult)
             .filter(TestResult.id.in_(tr_ids))
             .delete(synchronize_session=False))
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except OSError:
                pass
        db.delete(doc)  # cascades DocumentEntity + Chunk
        deleted += 1
    db.commit()
    return deleted
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_purge.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/services/purge.py tests/test_purge.py
git commit -m "feat(services): delete_documents (scoped purge + cascade)"
```

---

## Task 9: Delete route

**Files:**
- Modify: `app/api/routes_browse.py`
- Test: `tests/test_api_browse.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_api_browse.py`:

```python
def test_delete_records_endpoint():
    from app.db import SessionLocal
    from app.models import Document, Patient
    db = SessionLocal()
    try:
        p = Patient(name="Del Endpoint Person")
        db.add(p); db.flush()
        doc = Document(patient_id=p.id, doc_type="lab")
        db.add(doc); db.commit()
        pid, did = p.id, doc.id
    finally:
        db.close()
    r = client.post(f"/api/patients/{pid}/records/delete",
                    json={"document_ids": [str(did)]})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 1
```

(Uses the module-level `client` already defined at the top of `tests/test_api_browse.py`.)

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_browse.py::test_delete_records_endpoint -v`
Expected: FAIL — 404 (route not defined).

- [ ] **Step 3: Add the route to `app/api/routes_browse.py`**

Add to the schema import line: `DeleteRecordsIn`. Add the purge import near the other service imports:

```python
from app.services import purge as pgsvc
```

Extend the schema import (currently `from app.api.schemas import DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut`):

```python
from app.api.schemas import (
    DeleteRecordsIn, DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut,
)
```

Add the endpoint at the end of the file:

```python
@router.post("/patients/{patient_id}/records/delete")
def delete_records(patient_id: int, body: DeleteRecordsIn) -> dict:
    db = SessionLocal()
    try:
        n = pgsvc.delete_documents(db, patient_id, body.document_ids)
        return {"deleted": n}
    finally:
        db.close()
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_browse.py -v`
Expected: PASS (all browse tests green)

- [ ] **Step 5: Commit**

```bash
git add app/api/routes_browse.py app/api/schemas.py tests/test_api_browse.py
git commit -m "feat(api): POST /patients/{id}/records/delete"
```

---

## Task 10: Frontend API client + types

**Files:**
- Modify: `medagentic-dashboard/src/types.ts`
- Modify: `medagentic-dashboard/src/api.ts`

- [ ] **Step 1: Extend `ApiRecord` in `src/types.ts`** — add the three fields to the existing `ApiRecord` interface (after `description`):

```typescript
  value: string;
  unit: string;
  reference: string;
```

- [ ] **Step 2: Add `deleteRecords` to `src/api.ts`** — after `getDocuments`:

```typescript
export const deleteRecords = (patientId: string, documentIds: string[]) =>
  json<{ deleted: number }>(`/api/patients/${patientId}/records/delete`, {
    method: 'POST', body: JSON.stringify({ document_ids: documentIds }),
  });
```

- [ ] **Step 3: Type-check**

Run: `cd medagentic-dashboard && npx tsc --noEmit`
Expected: exit 0 (api.ts/types.ts clean; main.ts will use these next task).

- [ ] **Step 4: Commit**

```bash
git add medagentic-dashboard/src/types.ts medagentic-dashboard/src/api.ts
git commit -m "feat(ui): deleteRecords client + record value/unit/reference types"
```

---

## Task 11: Dated tables + delete in main.ts

**Files:**
- Modify: `medagentic-dashboard/src/main.ts`

Replace the flat card grid with per-date tables. Keep `esc()` on every dynamic string.

- [ ] **Step 1: Import `deleteRecords`** — extend the `from './api'` import block in `main.ts` to include `deleteRecords` (alphabetical or appended):

```typescript
import {
  createPatient, deleteRecords, getDocuments, getHealth, getRecords, listPatients,
  resumeChat, streamChat, uploadFile,
} from './api';
```

- [ ] **Step 2: Replace the `renderDashboard` records section** — in `renderDashboard`, replace everything from `const view = records` through the end of the `const grid = $('records-grid'); if (grid) { ... }` block with the grouped-table version:

```typescript
  const view = records.filter(r => filterType === 'all' || r.type === filterType);
  const groups = groupByDate(view, sortOrder);
  const grid = $('records-grid');
  if (grid) {
    grid.innerHTML = view.length === 0 ? `
      <div class="col-span-full text-center py-16 text-[#A6A298]">
        <i data-lucide="filter" class="w-10 h-10 mx-auto text-[#D5D2C9] mb-4"></i>
        <p class="text-lg font-light tracking-tight">No records found for this filter.</p>
      </div>` : groups.map(g => dateGroupHtml(g)).join('');
    bindDeleteButtons();
  }
```

- [ ] **Step 3: Add the grouping + table helpers** — add these functions just below `renderDashboard`:

```typescript
const DATE_COLORS = ['#5D7B6F', '#C16D54', '#6D6E9E', '#9E6D8A', '#6D9E97', '#9E946D'];

function dateColor(date: string): string {
  let h = 0;
  for (let i = 0; i < date.length; i++) h = (h * 31 + date.charCodeAt(i)) >>> 0;
  return DATE_COLORS[h % DATE_COLORS.length];
}

interface DateGroup { date: string; label: string; records: ApiRecord[]; docIds: string[]; }

function groupByDate(rows: ApiRecord[], order: 'desc' | 'asc'): DateGroup[] {
  const map = new Map<string, ApiRecord[]>();
  for (const r of rows) {
    const key = r.date ?? '';
    (map.get(key) ?? map.set(key, []).get(key)!).push(r);
  }
  const keys = [...map.keys()].sort((a, b) => {
    if (a === '') return 1;            // Undated last
    if (b === '') return -1;
    const t = new Date(a).getTime() - new Date(b).getTime();
    return order === 'desc' ? -t : t;
  });
  return keys.map(k => {
    const recs = map.get(k)!;
    const docIds = [...new Set(recs.map(r => r.id.split('-')[1]).filter(x => x && x !== 'undefined'))];
    return { date: k, label: k ? formatDate(k) : 'Undated', records: recs, docIds };
  });
}

function typeTag(t: string): string {
  return t === 'test_result' ? '' :
    `<span class="text-[9px] font-bold text-[#A6A298] uppercase tracking-widest mr-1.5">${esc(t.replace('_', ' '))}</span>`;
}

function dateGroupHtml(g: DateGroup): string {
  const color = g.date ? dateColor(g.date) : '#A6A298';
  const rows = g.records.map(r => `
      <tr class="border-t border-[#F0EFEB] hover:bg-[#FAF9F5]">
        <td class="py-2 px-3 text-[13px] font-semibold text-[#2E2C29]">${typeTag(r.type)}${esc(r.title)}</td>
        <td class="py-2 px-3 text-[13px] text-[#59554D] whitespace-nowrap">${esc([r.value, r.unit].filter(Boolean).join(' '))}</td>
        <td class="py-2 px-3 text-[12px] text-[#A6A298] whitespace-nowrap">${esc(r.reference || '—')}</td>
      </tr>`).join('');
  return `
    <div class="col-span-full mb-6 bg-white rounded-2xl border border-[#E0DDD5] shadow-sm overflow-hidden">
      <div class="flex items-center justify-between px-4 py-2.5" style="border-left:4px solid ${color}">
        <div class="flex items-center gap-2">
          <span class="w-2 h-2 rounded-full" style="background:${color}"></span>
          <span class="text-[13px] font-bold text-[#2E2C29]">${esc(g.label)}</span>
          <span class="text-[10px] text-[#A6A298] font-bold uppercase tracking-wider">${g.records.length} record${g.records.length === 1 ? '' : 's'}</span>
        </div>
        ${g.docIds.length ? `<button class="del-date text-[#C16D54] hover:text-[#a3553f] p-1" data-ids="${esc(g.docIds.join(','))}" data-label="${esc(g.label)}" data-count="${g.records.length}" title="Delete this date"><i data-lucide="trash-2" class="w-4 h-4"></i></button>` : ''}
      </div>
      <table class="w-full text-left">
        <thead><tr class="text-[10px] uppercase tracking-widest text-[#A6A298]">
          <th class="py-1.5 px-3 font-bold">Name</th>
          <th class="py-1.5 px-3 font-bold">Result</th>
          <th class="py-1.5 px-3 font-bold">Expected</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

function bindDeleteButtons() {
  document.querySelectorAll('.del-date').forEach(btn => {
    btn.addEventListener('click', async e => {
      const t = e.currentTarget as HTMLButtonElement;
      const ids = (t.dataset.ids || '').split(',').filter(Boolean);
      const label = t.dataset.label || 'this date';
      const count = t.dataset.count || ids.length;
      if (!ids.length) return;
      if (!confirm(`Delete all ${count} records from ${label}? This removes the document(s) and cannot be undone.`)) return;
      try {
        await deleteRecords(currentPatientId, ids);
        await loadPatientData();
        render();
      } catch (err: any) {
        banner(`Delete failed: ${err.message}`);
      }
    });
  });
}
```

- [ ] **Step 4: Type-check**

Run: `cd medagentic-dashboard && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add medagentic-dashboard/src/main.ts
git commit -m "feat(ui): per-date record tables, color-coded, with delete (esc XSS guard)"
```

---

## Task 11b: Editable HITL card (correct OCR before save)

**Files:**
- Modify: `medagentic-dashboard/src/main.ts`

The `confirm_ingest` card must let the reviewer edit extracted data before confirming.
Backend already persists `decision.extracted`, so this is frontend-only.

- [ ] **Step 1: Replace the `confirm_ingest` branch of `interruptCardHtml`** — swap the read-only summary for editable inputs. Replace the whole `if (payload.type === 'confirm_ingest') { ... }` block with:

```typescript
  if (payload.type === 'confirm_ingest') {
    const ex = payload.extracted || {};
    const name = ex.patient_name || '';
    const tests = ex.tests || [];
    const inp = (id: string, val: unknown, ph: string, cls = '') =>
      `<input id="${id}" value="${esc(val ?? '')}" placeholder="${esc(ph)}" class="${cls} bg-white border border-[#DFDDDA] rounded-md px-2 py-1 text-[12px] text-[#2E2C29] outline-none focus:border-[#5D7B6F]" />`;
    const testRows = tests.map((t: any, j: number) => `
        <div class="flex gap-1.5 items-center">
          ${inp(`int-t-${idx}-${j}-n`, t.name, 'test', 'flex-[2]')}
          ${inp(`int-t-${idx}-${j}-v`, t.value, 'value', 'flex-1')}
          ${inp(`int-t-${idx}-${j}-u`, t.unit, 'unit', 'w-16')}
          ${inp(`int-t-${idx}-${j}-r`, t.reference_range, 'ref', 'w-20')}
        </div>`).join('');
    const nameRows = (k: string) => (ex[k] || []).map((it: any, j: number) =>
      inp(`int-${k}-${idx}-${j}`, it.name, k, 'w-full')).join('');
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] tracking-widest uppercase">Human in the loop — edit then confirm</span>
        </div>
        <h3 class="text-xl font-light text-[#2E2C29] mb-4 tracking-tight">Verify &amp; Correct Extraction</h3>
        <div class="space-y-3 mb-5">
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-16">Patient</span>
            ${inp(`int-name-${idx}`, name, 'patient name', 'flex-1')}
          </div>
          <div class="flex gap-2 items-center">
            <span class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider w-16">Date</span>
            ${inp(`int-date-${idx}`, ex.doc_date, 'YYYY-MM-DD', 'flex-1')}
          </div>
          ${tests.length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Tests</div><div class="space-y-1.5">${testRows}</div></div>` : ''}
          ${(ex.diseases || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Diseases</div><div class="space-y-1.5">${nameRows('diseases')}</div></div>` : ''}
          ${(ex.symptoms || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Symptoms</div><div class="space-y-1.5">${nameRows('symptoms')}</div></div>` : ''}
          ${(ex.medications || []).length ? `<div class="pt-1"><div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider mb-1">Medications</div><div class="space-y-1.5">${nameRows('medications')}</div></div>` : ''}
        </div>
        <div class="flex gap-2.5">
          <button data-act="reject" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Reject</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Confirm &amp; Feed Layer</button>
        </div>
      </div>`;
  }
```

- [ ] **Step 2: Add a `collectExtracted` helper** — add just above `bindInterruptButtons`:

```typescript
function collectExtracted(idx: number, base: any): any {
  const ex = JSON.parse(JSON.stringify(base || {}));
  const g = (id: string) => ($(id) as HTMLInputElement | null)?.value;
  const nm = g(`int-name-${idx}`); if (nm !== undefined) ex.patient_name = nm;
  const dt = g(`int-date-${idx}`); if (dt !== undefined) ex.doc_date = dt;
  (ex.tests || []).forEach((t: any, j: number) => {
    const n = g(`int-t-${idx}-${j}-n`); if (n !== undefined) t.name = n;
    const v = g(`int-t-${idx}-${j}-v`); if (v !== undefined) t.value = v;
    const u = g(`int-t-${idx}-${j}-u`); if (u !== undefined) t.unit = u;
    const r = g(`int-t-${idx}-${j}-r`); if (r !== undefined) t.reference_range = r;
  });
  ['diseases', 'symptoms', 'medications'].forEach(k => {
    (ex[k] || []).forEach((it: any, j: number) => {
      const val = g(`int-${k}-${idx}-${j}`); if (val !== undefined) it.name = val;
    });
  });
  return ex;
}
```

- [ ] **Step 3: Use it in `bindInterruptButtons`** — collect inputs BEFORE the `chats.splice(idx, 1)` (splice + later render destroys the DOM). Replace the confirm-branch resume construction so it reads edited values:

```typescript
      let resume: any;
      if (payload.type === 'confirm_ingest') {
        resume = t.dataset.act === 'confirm'
          ? { approved: true, extracted: collectExtracted(idx, payload.extracted),
              ...(payload.patient_id ? { patient_id: payload.patient_id } : {}) }
          : { approved: false };
      } else {
        resume = { proceed: t.dataset.act === 'proceed' };
      }
      chats.splice(idx, 1); // remove the card (after reading its inputs)
```

(Ensure the existing `chats.splice(idx, 1)` that ran earlier in the handler is removed — it must happen AFTER `collectExtracted`.)

- [ ] **Step 4: Type-check**

Run: `cd medagentic-dashboard && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 5: Commit**

```bash
git add medagentic-dashboard/src/main.ts
git commit -m "feat(ui): editable HITL card - correct OCR/extraction before save"
```

---

## Task 12: Full run + self-review

**Files:** none (verification)

- [ ] **Step 1: Run the whole backend suite**

Run: `pytest -q`
Expected: all green (existing + new `test_dates`, `test_dedup`, `test_purge`, `test_entities_dates`, appended mapping/browse tests). If a pre-existing test was already failing before this work, note it but don't block.

- [ ] **Step 2: Frontend type-check**

Run: `cd medagentic-dashboard && npx tsc --noEmit`
Expected: exit 0.

- [ ] **Step 3: Manual smoke** — `make run` + `make ui`, open `http://localhost:3000`:
  1. Upload a dated PDF -> Confirm -> records appear under the document's **printed date** (not today), grouped in one dated table.
  2. Re-upload the SAME PDF -> chat says "already on file" instantly, **no** Verify-Extraction card.
  3. A date group's trash button -> confirm -> that date's rows disappear; documents list refreshes.
  4. Test rows show Name / Result (`value unit`) / Expected (`reference`) — reference no longer mixed into the result.

Expected: each behaves as described.

- [ ] **Step 4: Commit any doc/readme touch-ups if made** (else skip).

---

## Self-Review notes (reconciled against the spec)

- **Spec coverage:** report_date col + migration (T1); doc-date parse (T2); value/reference cleanup (T3,T6); persist report_date+observed_at (T4); browse prefers report_date (T5); early sha256 dedup before OCR/HITL (T7); delete service + route (T8,T9); frontend client + dated tables + delete (T10,T11); editable HITL card so the reviewer corrects OCR before save (T11b, frontend-only — backend already persists decision.extracted). Grouping by date + per-date color = T11. Forward-only (no backfill) and Tier-1-only (no simhash) honored — no tasks added for them.
- **Type consistency:** `parse_doc_date`/`format_value`/`delete_documents`/`deleteRecords` names match across tasks. `RecordOut` gains `value/unit/reference`; `ApiRecord` mirrors them. `dedup` state key values `"duplicate"/"new"` match the graph conditional. `DeleteRecordsIn.document_ids` matches the route + client body.
- **DB ordering:** migration applied in T1 before any persist/browse test (live-DB + create_all caveat noted in header).
- **Security:** every dynamic field in the new `main.ts` table/header HTML is `esc()`-wrapped; delete uses `confirm()`.
