# Medical Document Intelligence & Tracker

Foundation sub-project: a Python service/data layer on Supabase(Postgres/pgvector).
Runs locally. No UI yet.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`; set `DATABASE_URL` to your Supabase **session
   pooler** URL (IPv4: `postgresql+psycopg://postgres.<ref>:<pw>@aws-0-<region>.pooler.supabase.com:5432/postgres`).
4. `alembic upgrade head`

## Tests

`pytest -v`

DB tests run against the real Supabase Postgres (via `DATABASE_URL` /
`TEST_DATABASE_URL`) so pgvector behaviour is exercised, not mocked.

## Notes

- Direct Supabase connections are IPv6-only; use the session pooler URL for
  IPv4 networks and tooling (Alembic, local runs).
- Raw files are stored on local disk under `STORAGE_DIR`; the DB stores only
  the path.

## Build order

See `.claude/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
