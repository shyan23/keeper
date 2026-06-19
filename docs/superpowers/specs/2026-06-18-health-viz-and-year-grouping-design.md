# Health Visualization + Year-Grouped Records — Design

Date: 2026-06-18
Branch: feat/agentic_chatbot

## Problem

Doctors track a patient's health across years. Today the dashboard shows:

- An **All** tab = a flat, ungrouped document table. With years of reports it
  becomes a long undifferentiated list.
- Per-entity tabs (disease / symptom / medicine / test_result) grouped by
  document, but **no trend over time** — a doctor cannot see whether a value
  (e.g. hemoglobin) is rising or falling across visits.

Goal: (1) visualize numeric test-result trends over time, and (2) reorganize the
All tab so reports are grouped by **year**, and within a year by **report
category** (Hematology, X-Ray, Urine…).

## Decisions (locked during brainstorming)

- **Visualization** = test-result **trend line charts** (value vs date per metric).
- **All tab** = **Year → category → documents** (expandable).
- **Category** = fine-grained, **LLM-extracted from the report itself** (the PDF
  prints it: "Hematology", "X-Ray Report", "Urine R/E", "Lipid Profile"…).
  Stored in the existing-but-unused `Document.classification` column.
- **Chart rendering split, one data source of truth:**
  - A backend **trend-series service** computes the numeric series (pure data).
  - **Now — Chart.js** renders that series in a new **Trends** tab (interactive,
    instant metric switching, no per-switch roundtrip).
  - **Later — matplotlib** reads the *same* series service to render PNGs for the
    future PDF report. Not built in this feature (YAGNI), but the series service
    is designed so it drops in unchanged.
- **Reference band**: shade the normal range behind the line; color out-of-range
  points red.

## Architecture

Four independent units. Each ships and is testable on its own.

```
┌─────────────────────────────────────────────────────────────┐
│ 1. LLM category extraction (segment.py / ingest.py)          │
│    ReportSpec gains `category` → stored in Document.classification │
└───────────────┬─────────────────────────────────────────────┘
                │ classification column populated
                ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Trend-series service (app/services/trends.py)             │
│    list_metrics(patient) -> [{key,label,unit,n}]             │
│    metric_series(patient, key) -> {points[], ref_low, ...}   │
└───────┬──────────────────────────────┬──────────────────────┘
        │ JSON via routes_browse.py     │ (future) matplotlib PNG
        ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│ 4. Trends tab        │        │ (future) PDF report  │
│    Chart.js          │        │    same series fn     │
└──────────────────────┘        └──────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. Year-grouped All tab (main.ts documentsTableHtml rewrite) │
│    uses Document.classification + report_date                │
└─────────────────────────────────────────────────────────────┘
```

---

## Unit 1 — LLM category extraction

**What it does:** during report segmentation, capture each report's printed
category and persist it for grouping.

**Changes:**

- `app/services/segment.py`
  - `ReportSpec` gains a field: `category: str | None = None` — "the report's
    department/panel name as printed (e.g. Hematology, Biochemistry, X-Ray,
    Ultrasound, Urine, Lipid Profile). null if not stated."
  - Extend `_SPLIT_PROMPT` to request `category` per report.
  - `split_reports` output dict gains `"category"`. Regex fallback sets it to
    `None` (no reliable signal without the LLM).
- `app/agent/nodes/ingest.py`
  - Carry `seg["category"]` through to the document-creation call.
- `app/services/documents.py`
  - `create_document(...)` gains `classification: str | None = None`, written to
    `Document.classification`.

**Backfill:** existing documents have `classification = NULL`. They render under
an **"Uncategorized"** group until re-ingested. No migration backfill in this
feature — acceptable because the column already exists (nullable). A one-off
re-classify script is out of scope (note it as future work).

**Depends on:** the LLM split path. The regex fallback yields no category.

---

## Unit 2 — Trend-series service

**File:** `app/services/trends.py` (new). Pure functions on a DB session — no
HTTP, no rendering. This is the shared source of truth for Chart.js now and
matplotlib later.

**Metric identity:** a metric = a normalized test name. Group `test_result`
records by `lower(trim(test))`. `label` = the most common original casing.
Only expose metrics with **≥ 2 numeric data points** (a single point is not a
trend).

```python
def list_metrics(db, patient_id) -> list[dict]:
    # [{ "key": "hemoglobin", "label": "Hemoglobin", "unit": "g/dL", "n": 5 }, ...]
    # only metrics whose value parses to a number on ≥2 dated reports

def metric_series(db, patient_id, key) -> dict:
    # {
    #   "key": "hemoglobin", "label": "Hemoglobin", "unit": "g/dL",
    #   "ref_low": 12.0, "ref_high": 16.0,   # parsed from reference, may be null
    #   "points": [ { "date": "2024-03-04", "value": 11.2, "in_range": false }, ... ]
    # }
    # points sorted ascending by date; undated points excluded
```

**Value parsing:** reuse the numeric-normalization already in
`app/api/mapping.format_value`; extract the leading float. Non-numeric values
(e.g. "Positive", "Trace") are skipped — they cannot be plotted.

**Reference parsing:** parse `low-high` from the record's `reference` string
(the `_REF_RE` pattern in mapping.py). If unparseable, `ref_low/ref_high = null`
and all points are `in_range: true` (no band, no red).

**Depends on:** existing `bsvc.list_test_results` + mapping parsing helpers.

---

## Unit 3 — Year-grouped All tab

**File:** `medagentic-dashboard/src/main.ts` — rewrite `documentsTableHtml`
(and supporting group logic). No backend change beyond exposing
`classification` on the documents endpoint.

**Backend:** `app/api/mapping.document_to_out` adds
`"category": row.get("classification")` and `routes_browse.patient_documents` /
`browse.list_documents_timeline` select the `classification` column.
`types.ts ApiDocument` gains `category: string | null`.

**Grouping logic (frontend):**

1. Bucket documents by **year** of `report_date` (fall back to upload `date`;
   undated → an "Undated" bucket sorted last).
2. Within a year, bucket by `category` (fall back to `type`/doc_type, then
   "Uncategorized").
3. Sort years by `sortOrder` (respects the existing Newest/Oldest toggle).

**Render:** expandable year rows (reusing the existing `expandedCards` +
`card-toggle` machinery and `tableShell`).

```
▾ 2025                               8 reports
   Hematology (3)
     • CBC report        Mar 4 2025   [open] [del]
     • Blood panel       Jun 1 2025   [open] [del]
   X-Ray (2)
     • Chest X-Ray       Feb 9 2025   [open] [del]
   Urine (3)
     ...
▸ 2024                              12 reports
▸ Undated                            1 report
```

- Year row = toggle button (chevron, year, total report count).
- Category sub-header = small uppercase label + count.
- Document row keeps current affordances: colored dot, name (opens PDF),
  date, open + delete actions.

**Depends on:** Unit 1 for meaningful categories (degrades to "Uncategorized").

---

## Unit 4 — Trends tab (Chart.js)

**Frontend:**

- Add **Chart.js** dependency (`npm i chart.js`).
- Add a **"Trends"** entry to the filter bar between "All" and "disease":
  `['all', 'trends', 'disease', 'symptom', 'medicine', 'test_result']`.
- New API client fns in `api.ts`:
  `getTrendMetrics(patientId)` and `getTrendSeries(patientId, key)`.
- When `filterType === 'trends'`:
  - Fetch metric list; render a metric `<select>` dropdown (default = first
    metric, or remembered selection in a module-level `trendMetric` var).
  - Fetch the selected series; render a Chart.js line chart in a `<canvas>`:
    - line of `value` vs `date`;
    - shaded band between `ref_low`/`ref_high` (Chart.js `fill` between two
      datasets, or an annotation-style band drawn as a filled dataset);
    - out-of-range points (`in_range === false`) styled red.
  - Empty state when the patient has no metric with ≥2 numeric points.
- Destroy/replace the Chart instance on metric switch and patient switch to
  avoid canvas leaks.

**Backend routes** (`routes_browse.py`, new):

```
GET /api/patients/{id}/trends            -> list[MetricOut]
GET /api/patients/{id}/trends/{key}      -> SeriesOut
```

Pydantic schemas `MetricOut`, `SeriesOut`, `SeriesPointOut` in `schemas.py`,
both delegating to `app/services/trends.py`.

**Depends on:** Unit 2.

---

## Data flow (Trends tab)

```
patient select → GET /trends → [metrics]
   → dropdown render (pick metric)
   → GET /trends/{key} → series JSON
   → Chart.js render (line + band + red points)
metric switch → GET /trends/{key} → re-render (no full reload)
```

## Error handling

- No numeric metrics → Trends empty state ("No trend data yet — needs ≥2
  numeric results for a test").
- Unparseable reference → plot line only, no band, no red points.
- Undated test results → excluded from series (cannot place on time axis).
- Series/metrics fetch failure → inline error in the tab, dashboard otherwise
  unaffected (same `.catch(() => [])` pattern as existing fetches).
- LLM omits category → document falls into "Uncategorized" (All tab still works).

## Testing

- **Unit 2 (pytest):** `list_metrics` excludes single-point and non-numeric
  metrics; `metric_series` sorts by date, flags out-of-range points, parses /
  fails-soft on reference ranges, excludes undated points.
- **Unit 1 (pytest):** `split_reports` surfaces `category`; `create_document`
  persists `classification`; regex fallback → `category=None`.
- **Routes (pytest):** `/trends` and `/trends/{key}` shapes; 404 on unknown
  patient; documents endpoint includes `category`.
- **Frontend (existing vitest harness):** year+category grouping buckets
  correctly (years sorted by sortOrder, undated last, uncategorized fallback);
  trends dropdown + canvas mount; Chart instance replaced on switch.

## Out of scope (future)

- matplotlib PNG renderer + the PDF report itself (reads the same series fn).
- Re-classify backfill script for already-ingested documents.
- Multi-metric overlay / report-volume bar chart / vitals sparkline grid.
```
