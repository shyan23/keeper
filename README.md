# Medical Document Intelligence & Tracker

Python service/data layer on Supabase (Postgres/pgvector) + LangGraph agent.
FastAPI backend, Vite/TypeScript dashboard. Runs locally.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`; set `DATABASE_URL` to your Supabase **session
   pooler** URL (IPv4: `postgresql+psycopg://postgres.<ref>:<pw>@aws-1-<region>.pooler.supabase.com:5432/postgres`;
   new projects use the `aws-1-<region>` host, not `aws-0`). Set `TEST_DATABASE_URL`
   to a **separate** throwaway DB — never the same as `DATABASE_URL` (see Tests).
4. `alembic upgrade head`

## Run

`make` targets wrap the commands:

    make run    # backend  (FastAPI :8000, single worker)
    make ui     # frontend (Vite :3000)
    make dev    # both (backend backgrounded; Ctrl-C stops UI)

Raw equivalents:

Backend (single worker — agent checkpointer is in-process memory):

    uvicorn app.api.server:app --port 8000 --workers 1

Frontend (Vite dev server on :3000):

    cd medagentic-dashboard
    npm install
    npm run dev

Set `VITE_API_BASE` in `medagentic-dashboard/.env` if API not on `http://localhost:8000`.
Frontend typecheck: `cd medagentic-dashboard && npx tsc --noEmit`.

## Tests

    pytest -v

DB tests exercise the real Postgres/pgvector (not mocked).

**WARNING — test DB must NOT be the prod DB.** The autouse fixtures delete
rows; pointing tests at `DATABASE_URL` wipes real patient data. `TEST_DATABASE_URL`
must be set to a **separate throwaway database** — `tests/conftest.py` refuses to
run (raises at collection) when it is empty or equal to `DATABASE_URL`. Cleanup
removes only test-created patients (snapshot diff), never pre-existing data.

## Fixes (fix/UI)

Caveman log. What change:

- **Dates.** OCR doc date now SAVED on document (`report_date`, mig 0004 + 0003). Was dropped before. Show report date everywhere — docs list, citations — not upload time.
- **Citations.** No more `#chunk` ids + raw OCR glued in answer. Answer = clean text. Sources = chip per document (type + date). Click chip → open original file new tab.
- **File serve.** `GET /api/documents/{id}/file` stream original PDF/image. Absolute storage paths → survive restart / cwd change.
- **Doc name.** Store uploaded filename (`original_name`). No more `12.pdf`.
- **Patient pick.** Query name no patient → agent ask which patient. Autocomplete card in chat. Resume with chosen id.
- **Fuzzy name.** Slight name diff on ingest → ask "same person as X?" (difflib ratio ≥ 0.85). No silent dup profile.
- **Docs tab.** Table: name / type / date / Open. Search + type filter + sort (newest/oldest/type). Scale 100+. "Active Context" gone.
- **Ingestion stepper.** Vertical Upload → OCR → Extract → Review → Index. Spinner active, check done. Kill "OCR 1/4" line.
- **Resizable divider.** Drag dashboard ↔ chat panel. Width persist (localStorage).

### fix/UI v2 — document-centric records + chat edits

- **"All" tab = documents only.** Card per uploaded PDF (name/type/OCR date). Click → open PDF. No entity dump. Delete per card.
- **Entity tabs = expandable cards.** Disease / Symptom / Medicine / Test result: one card per source document, headed by OCR date. Expand → that doc's items; test_result expands to full result table (Test / Result / Expected). "View source document" link inside.
- **Knowledge panel slimmed.** Dropped duplicate docs table. Panel just stages uploads; files show in the All tab.
- **OCR date fallback.** LLM `doc_date` null → scrape report date from OCR text (`services/dates.date_from_text`). No more today/upload-date.
- **Latest-value answers.** RAG orders snippets newest-first + tags each with date. "what is the RBC?" → most recent value + its date. Older/multiple only when query names a date / period / trend.
- **Honorific-aware naming.** Match strips titles (MRS./Dr./Ms…) before exact + fuzzy. `MRS. NAFISA KABIR` == `NAFISA KABIR` → one person; ambiguous tie → "same person as X?" gate.
- **Dedup actually fires.** Each upload gets a fresh LangGraph thread, so stale ingest state (document_id, already_ingested…) can't bleed in. 2nd+ PDFs persist; re-upload same file → "already on file".
- **Edit extracted data via chat (HITL).** e.g. "set hemoglobin to 1.2", "correct the report date to 5 Oct 2023". `router → plan_edit → confirm_edit`. plan_edit locates the newest matching record (fuzzy: `haemoglobin level` → `Hemoglobin (Hb%)`, British/American + filler tolerant), then a **verify card** shows current → new (editable). Confirm writes; Cancel does nothing.
- **Structured query shows the doc.** "show me the latest document" → clickable document chip (open PDF). Patient filter honorific-tolerant; no name → scopes to the selected patient.

## Notes

- Direct Supabase connections are IPv6-only; use the session pooler URL for
  IPv4 networks and tooling (Alembic, local runs).
- Raw files are stored on local disk under `STORAGE_DIR` (absolute path); the DB
  stores the path + original filename. Serve via `GET /api/documents/{id}/file`.

## Build order

See `.claude/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
