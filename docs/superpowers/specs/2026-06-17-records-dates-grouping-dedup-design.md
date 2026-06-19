# Records: real dates, dated grouping, clean values, early dedup — Design

**Date:** 2026-06-17
**Branch:** feat/agentic_chatbot
**Depends on:** the UI↔backend integration (FastAPI + vanilla-TS dashboard) already landed on this branch.

## Problem

Three defects observed in the live dashboard:

1. **Wrong dates.** Every record shows the upload date (today), not the date printed
   on the document. Example: `CamScanner 16-6-26 11.40.pdf` is dated **Oct 5, 2023**;
   the dengue test reads **05/10** — those are the dates that matter clinically.
   The LLM already extracts `doc_date`, but `persist_extraction` never stores it and
   the browse queries read `Document.uploaded_at`.

2. **Reference range leaks into the value.** `TestResult.reference_range` is stored in
   its own column, but the record card dumps the raw `source_span` (e.g. `"00-05 %"`)
   as the body and ignores the clean fields, so the reference interval appears mixed
   with the result. Values are also unformatted.

3. **Records aren't grouped.** A flat grid mixes every test/disease/med from every
   document and date. With multiple documents there's no way to see "all results from
   the Oct 5 report" together.

4. **Dedup fires too late.** Re-uploading the *same* PDF re-runs OCR, the extraction
   LLM, and the human-in-the-loop gate before the `content_hash` check (buried in
   `create_document_node`) ever runs. The duplicate should be caught from the file
   bytes, before any of that.

## Decisions (locked with the user)

- **Grouping:** by report date — one section per date, newest first. Two documents on
  the same calendar date merge into that date's section.
- **Color:** accent color per report date (each date gets its own color), deterministic
  so the same date renders the same color every time.
- **Date storage:** add `Document.report_date`; derive all record dates from it.
- **Backfill:** forward-only. Existing rows keep showing upload date until re-uploaded.
- **Dedup:** Tier 1 only — exact `sha256` of the file bytes, checked before OCR. (No
  simhash/near-duplicate tier.)

## Design

### 1. Capture the document date (backend)

- **Migration `0003`:** add `document.report_date DATE NULL`.
- **Date parsing helper** `app/services/dates.py::parse_doc_date(s) -> date | None`.
  Handles the common shapes the OCR/LLM emits: `2023-10-05`, `05/10/2023`,
  `5 Oct 2023`, `October 5, 2023`, `05-10-23`. Ambiguous `dd/mm` vs `mm/dd` resolves
  **day-first** (the documents are Bangladeshi lab reports). Unparseable → `None`.
- **`persist_extraction`** (and the ingest persist path): parse `result.doc_date`,
  set `Document.report_date`, and set `TestResult.observed_at` from the same parsed
  date (column already exists). `persist_extraction` gains the `document` so it can set
  `report_date` once per ingest.

### 2. Surface the date (browse queries)

- `list_test_results` and `list_entity_links` select `Document.report_date` and return
  `date = report_date or uploaded_at` (ISO `YYYY-MM-DD`). Old docs (no report_date)
  still show their upload date; new docs show the true date.
- Ordering switches to `report_date` desc (fallback uploaded_at) so the newest *report*
  sorts first, not the newest upload.

### 3. Clean values (mapping)

`app/api/mapping.py::merge_records`:
- Carry `unit` and `reference_range` as **separate** fields on the record dict
  (`RecordOut` gains `unit` and `reference` optional fields).
- `title` for a test = the test name (e.g. `HbA1c`). Body = formatted `value unit`.
- **Value formatting** `format_value(value, unit)`:
  - strip surrounding whitespace;
  - if a reference interval got glued onto the value (e.g. `"52  0-15"`,
    `"6.8 (4.0-6.0)"`), split it off — keep the leading number(s) as the value, move the
    interval to `reference` only if `reference_range` is empty;
  - normalize numbers (`".01"`→`"0.01"`, drop trailing `.0`), leave non-numeric values
    (`"Negative"`, `"Positive"`) untouched.
- `source_span` no longer renders as the card body.

### 4. Group + table (frontend `main.ts`)

No more card-per-record (100 tests must not be 100 boxes). Render a **compact table per
date group**.

- `renderDashboard` groups the filtered records by `date` → an ordered list of
  `{ date, records[] }`, newest first; `null`/empty dates fall into an "Undated" group
  last.
- Each group renders a slim **header row** — the date (`Oct 5, 2023`) + doc type, with a
  small accent dot/left-border colored deterministically per date (stable hash → fixed
  palette index) and a trash button (section 6) — followed by a **table**:

  | Name | Result | Expected |
  |------|--------|----------|
  | tests | `value unit` | `reference` |
  | disease/symptom/medicine | name | — (Result/Expected blank) |

  Columns: **Name**, **Result** (`value unit` for tests; the entity name doubles as Name
  so Result is blank for non-tests), **Expected** (`reference` for tests, else blank). A
  tiny type tag in the Name cell keeps disease/symptom/medicine distinguishable.
- Dense rows, minimal chrome — borders + zebra at most, no per-row shadows/padding-heavy
  boxes. The table scrolls inside the records pane.
- Type filter chips and newest/oldest sort still apply (sort reorders the date groups;
  filter narrows the rows; an empty group is hidden).
- All dynamic strings stay wrapped in `esc()` (XSS guard).

### 5. Early dedup (ingest graph)

- **New node `dedup_check`** inserted after `router` (ingest branch) and **before**
  `extract_text`:
  - read staged bytes, compute `sha256`, `find_by_content_hash`;
  - **hit:** set `already_ingested=True`, reuse `document_id`/`patient_id`, delete the
    staged file, and route to the terminal "already on file" reply — **skipping OCR,
    extraction, and the HITL gate**;
  - **miss:** pass the computed `content_hash` forward and continue to `extract_text`.
- **Graph wiring:** conditional edge out of `dedup_check`: `duplicate → END` (after the
  skip message), `new → extract_text`. `extract_text_node` no longer needs to compute the
  hash (it's already in state); it keeps working if it's still set there.
- `create_document_node`'s existing hash check stays as a defensive backstop (harmless;
  the early node makes it rarely reached on dupes).

### 5b. Editable HITL card (correct OCR before saving)

OCR/LLM extraction is imperfect; the human gate must let the reviewer **fix** the
extracted data before it's committed, not just approve/reject.

- The backend already supports this: `confirm_ingest_node` persists
  `decision.get("extracted", ex)`, so whatever `extracted` the frontend returns on
  resume is what gets saved. **No backend change** beyond what's already there.
- Frontend only: the `confirm_ingest` card renders the extracted data as **editable
  inputs** — patient name, and per test row `name / value / unit / reference`, plus
  disease/symptom/medication names. On **Confirm**, rebuild the `extracted` dict from the
  current input values and send it as `resume.extracted` (existing patient-id passthrough
  unchanged). Reject path unchanged.
- All input values are read via `.value` (no innerHTML round-trip), so no new XSS surface;
  rendered defaults still pass through `esc()`/attribute-encoding.

### 6. Delete a date's records (backend + UI)

Need to erase wrong/old data. Because the UI groups by date, deletion is per date
group — but keyed on the **document ids** in that group (the records already carry
`document_id`), so it works for the "Undated" group and fallback-dated old data too.

- **Service** `app/services/purge.py::delete_documents(db, patient_id, document_ids)`:
  - scope-check every id belongs to `patient_id` (ignore/skip foreign ids — never delete
    across patients);
  - delete the `TestResult` rows referenced by those documents' `test_result` links
    (TestResult has no FK to Document, so it won't cascade on its own);
  - delete the `Document` rows — `DocumentEntity` and `Chunk` cascade via their
    `ondelete="CASCADE"` FKs;
  - delete the backing files on disk (`file_path`), ignoring missing files;
  - leave the shared name tables (`Disease`/`Symptom`/`Medication`/`MedicalTest`)
    untouched — other documents may reference them.
  - returns the count of documents deleted.
- **Endpoint** `POST /api/patients/{patient_id}/records/delete` with body
  `{ "document_ids": ["3","4"] }` → `{ "deleted": <n> }`. POST (not DELETE) so the body
  is unambiguous across clients.
- **Frontend:** each date-group header gets a small trash button. Click → a confirm
  ("Delete all N records from <date>? This removes the document(s) and cannot be
  undone.") → collect the group's distinct `document_id`s → call the endpoint → reload
  records + documents. Confirmation is mandatory (destructive, irreversible).

This is the path to wipe the current bad data: each old record shows today's upload date,
so they collapse into one group — delete it to clear them, then re-upload for correct
dates.

## Testing (TDD)

- `parse_doc_date`: table of input strings → expected `date`/`None`, incl. day-first.
- `format_value`: glued-reference split, number normalization, non-numeric passthrough.
- `persist_extraction`: sets `Document.report_date` and `TestResult.observed_at` from
  `doc_date`; `None` when absent/unparseable.
- browse: a doc with `report_date` returns that date and sorts before an older-reported
  doc uploaded later.
- `merge_records`: emits separate `unit`/`reference`, formatted value, no source span in
  body.
- dedup: a second ingest of identical bytes hits `dedup_check` → `already_ingested`, and
  the fake graph never reaches `extract_text`/`confirm_ingest`.
- frontend: `tsc --noEmit` clean; grouping helper unit-tested if extracted as a pure fn.
- delete: `delete_documents` removes the docs + their TestResults + cascades
  DocumentEntity/Chunk, leaves shared name tables, skips foreign-patient ids; the
  endpoint returns the deleted count and the records/documents go empty afterward.

## Out of scope

- Near-duplicate (simhash) detection.
- Backfilling report dates onto existing rows.
- Abnormal-value (out-of-range) color coding.
- Re-OCR of already-ingested documents.
