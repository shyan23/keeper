# PDF Report Generation — Design

**Date:** 2026-06-19
**Branch:** `feat/pdf-generation`
**Status:** Approved

## Goal

Generate a downloadable, professional medical PDF from a natural-language request
("make a pdf of all lipid profile results of patient X for the last 3 years"). The
agent resolves patient + timeframe + document types, aggregates already-extracted
medical data, renders trend charts, assembles a sectioned PDF with the original
source documents appended, and delivers a download link — with two human approval
gates.

## Key decisions

1. **Data source = existing DB, not re-extraction.** Diseases, symptoms, tests, and
   measurements are already extracted and stored at ingest (the user already reviewed
   them at that gate). The PDF pipeline *queries and aggregates* this data. No OCR, no
   LLM extraction at PDF time → fast, $0, deterministic, reproducible, and consistent
   with what the user confirmed. The only LLM call is parsing the NL request into
   structured filters.
2. **Two HITL gates, not six.** Consolidates the spec's six stages into:
   - **Gate A — Plan** (interpretation + discovery): resolved patient, timeframe,
     included/excluded documents. Approve / Modify.
   - **Gate B — Delivery**: built PDF preview, page/section/attachment summary,
     download link. Download / Regenerate.
   This matches the codebase's existing single-gate-per-workflow style
   (`confirm_ingest`, `confirm_edit`) and keeps extraction/charting/assembly automatic
   between the gates.
3. **Attachments included.** Every source document within the timeframe is appended
   (PDFs merged via `insert_pdf`, images as full pages), with an appendix index.
4. **PDF engine = pymupdf (already installed) + matplotlib (new).** No reportlab /
   weasyprint. fitz `Story` renders reflowable HTML for the body sections; charts are
   matplotlib PNGs embedded as image pages; `Document.insert_pdf()` appends source
   PDFs. matplotlib is the only new dependency.

## Architecture

New intent `generate_pdf` joins the router. Four LangGraph nodes:

```
router → plan_report ──▶ confirm_report (GATE A) ──▶ build_report ──▶ deliver_report (GATE B) ──▶ END
              │ (no patient / no docs)                                       │ regenerate
              └────────────────▶ END                                        └──▶ build_report
```

- `plan_report_node` — parse request → `PdfRequest`; resolve patient (named, else
  UI-selected `patient_id`); resolve timeframe; gather *candidate* documents (counts,
  included/excluded by date-window + fuzzy doc_type). Dead-ends to END with a message
  if no patient or no matching docs.
- `confirm_report_node` — `interrupt({type: "confirm_report", plan})`. Gate A.
  Approve → continue. Modify → resume carries edited timeframe / doc_types → re-plan.
  Reject → END.
- `build_report_node` — full `gather()`, render charts, assemble PDF, write to storage,
  set `report_path` / `report_url`. Emits progress labels (gather → charts → assemble →
  attachments) via the existing `progress` callback / stepper.
- `deliver_report_node` — `interrupt({type: "confirm_delivery", summary, url})`. Gate B.
  Download = terminal approval (final assistant message with link). Regenerate → loop
  back to `build_report`.

## New files (each one responsibility)

| File | Responsibility |
|---|---|
| `app/services/report.py` | Pure functions over a DB session. `parse_request(deps, text) -> PdfRequest` (LLM structured: `patient_name`, `doc_types[]`, `years[]`, `last_n_years`, `last_n_months`). `resolve_timeframe(req, today) -> (date_from, date_to)`. `gather(db, patient_id, doc_types, date_from, date_to) -> ReportData` aggregating documents, diseases, symptoms, tests, most-recent age, timeline, attachment list. Reuses `browse.list_entity_links` / `list_test_results` / `list_documents_timeline`, `structured._doc_matches`, `dates.parse_doc_date`. |
| `app/services/charts.py` | matplotlib `Agg` backend. `render_metric_chart(series) -> bytes` (PNG) per numeric metric, fed by `trends.metric_series` (chronological points, axis labels, units, reference band). |
| `app/services/pdf.py` | `build_report(data, charts, attachments) -> bytes` via pymupdf. fitz `Story` HTML for the 8 body sections; chart pages via `page.insert_image`; `doc.insert_pdf()` / image pages to append originals; appendix index. |
| `app/agent/nodes/report.py` | `plan_report_node`, `confirm_report_node`, `build_report_node`, `deliver_report_node`. |

## Touched files

- `app/agent/router.py` — add `generate_pdf` label + classifier guidance.
- `app/agent/graph.py` — register 4 nodes; `router` conditional edge `generate_pdf →
  plan_report`; plan→confirm (or END), confirm→build (or END), build→deliver,
  deliver→build (regenerate) / END.
- `app/agent/state.py` — `AgentState` keys: `report_request`, `report_plan`,
  `report_path`, `report_url`.
- `app/storage.py` — `save_report(data: bytes) -> str` under `STORAGE_DIR/_reports/`.
- `app/api/routes_chat.py` — `GET /api/chat/report/{name}` → `FileResponse`
  (`application/pdf`, content-disposition attachment). Name is the stored uuid filename;
  reject path traversal.
- `app/api/runtime.py` — `NODE_LABELS` for the 4 nodes.
- `medagentic-dashboard/src/main.ts` — two interrupt cards reusing existing
  `interruptCardHtml` / resume machinery: `confirm_report` (patient, timeframe,
  included + excluded files → Approve / Modify), `confirm_delivery` (page/section/
  attachment summary, Download link, Regenerate). Progress stepper reused for the build
  phase.
- `requirements.txt` — add `matplotlib`.

## PDF layout (spec section order)

1. Cover page (patient name prominent, timeframe, generated date)
2. Patient information (name, most-recent age, gender)
3. Disease summary (deduped, diagnosis chronology, evidence-backed only)
4. Symptoms summary (per disease where available; omit if none)
5. Medical test results (value, unit, reference range, collection date, source)
6. Charts & trends (matplotlib time series per metric)
7. Timeline of findings (documents/events by date)
8. Attached original documents (+ appendix index)

## Age resolution ("most recent age")

`Patient.age` stores a single value, not a per-document snapshot. Resolve the
most-recent age by regex-scanning `raw_ocr_text` of the newest in-window dated document
for an age pattern; fall back to `Patient.age`. `ponytail:` comment marks the ceiling —
upgrade to a stored per-document age if accuracy demands it. Filesystem / upload
timestamps are never used for age.

## Quality rules (from spec)

- No fabrication: PDF contains only DB-stored, provenance-tracked data.
- Provenance (source span, document id, date) carried on every aggregated row.
- Document dates honored; `uploaded_at` used only as a date fallback, never for age.
- Deterministic `gather` + `resolve_timeframe` → reproducible output.

## Out of scope (deliberate simplifications)

- **Estimated-completion timer.** Stage labels via the existing stepper suffice; build
  is seconds. Add a timer only if generation grows slow.
- **Cancel-mid-generation.** Build is a few seconds of synchronous work; thread-cancel
  plumbing isn't worth it.
- **Six separate approval gates.** Consolidated to two (see decisions).
- **Re-extraction at PDF time.** Reuses confirmed DB data.

Add any of these back if required.

## Testing (TDD)

Pure functions and nodes get tests with fake deps / pgvector test DB:

- `resolve_timeframe`: last-N-years, last-N-months, explicit year list, no constraint.
- `gather`: in/out-of-window filtering, doc_type fuzzy match, age resolution, empty case.
- `charts.render_metric_chart`: returns non-empty PNG bytes with valid header.
- `pdf.build_report`: returns valid PDF; expected page count; source PDFs appended;
  image attachment becomes a page.
- nodes: `plan_report` dead-ends (no patient / no docs); `confirm_report` approve /
  modify / reject; `deliver_report` download / regenerate loop.
