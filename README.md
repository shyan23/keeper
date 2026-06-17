# Medical Document Intelligence & Tracker

Python service/data layer on Supabase (Postgres/pgvector) + LangGraph agent.
FastAPI backend, Vite/TypeScript dashboard. Runs locally.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`; set `DATABASE_URL` to your Supabase **session
   pooler** URL (IPv4: `postgresql+psycopg://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres`).
4. `alembic upgrade head`

## Run

Backend (FastAPI, single worker — agent checkpointer is in-process memory):

    uvicorn app.api.server:app --port 8000 --workers 1

Frontend (Vite dev server on :3000):

    cd medagentic-dashboard
    npm install
    npm run dev

Set `VITE_API_BASE` in `medagentic-dashboard/.env` if API not on `http://localhost:8000`.

## Tests

`pytest -v`

DB tests run against the real Supabase Postgres (via `DATABASE_URL` /
`TEST_DATABASE_URL`) so pgvector behaviour is exercised, not mocked.

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

## Notes

- Direct Supabase connections are IPv6-only; use the session pooler URL for
  IPv4 networks and tooling (Alembic, local runs).
- Raw files are stored on local disk under `STORAGE_DIR` (absolute path); the DB
  stores the path + original filename. Serve via `GET /api/documents/{id}/file`.

## Build order

See `.claude/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
