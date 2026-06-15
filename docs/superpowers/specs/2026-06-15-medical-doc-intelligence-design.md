# Medical Document Intelligence & Tracker — Design

Date: 2026-06-15
Status: Approved (overall shape). Foundation sub-project detailed below.

## Vision

Multi-profile medical record system. Ingests prescriptions, lab reports, and
medical documents; extracts structured medical knowledge; serves a strictly
grounded RAG chatbot that answers only from uploaded documents, with citations.

## Locked Decisions

| Layer | Choice |
|---|---|
| Backend | Python + FastAPI (hosted). Also serves the static frontend. |
| Frontend | Vanilla HTML / CSS / JS. No framework. |
| AI provider | Google Gemini API only — vision OCR, entity extraction, `text-embedding-004` embeddings, chat (`gemini-2.0-flash`). |
| Database | Supabase (hosted PostgreSQL) + pgvector. Holds structured entities AND embeddings. |
| File storage | Raw images/PDFs on local disk. DB stores only file paths + metadata. |
| Auth | None. Single owner manages multiple patient profiles. Isolation enforced at query level. |
| Topology | One repo, one FastAPI monolith (API + static UI). |

## Decomposition (build order)

Each sub-project gets its own spec → plan → build cycle.

1. **Foundation** — repo scaffold, Postgres schema + migrations, file-storage
   layout, FastAPI skeleton, config/secrets, health check. *(this spec)*
2. **Ingestion pipeline** — upload → save file → Gemini OCR → classify →
   extract entities → persist + link to document.
3. **RAG system** — chunk → embed → pgvector retrieval → rerank → grounded
   answer with citations + patient/disease/date filters + cross-patient isolation.
4. **UI** — the 5 screens against the API.
5. **Human-in-loop** — edit entities, fix OCR text, merge duplicate entities,
   validate classifications.
6. **Organization layer** — per-patient / per-disease grouping, timeline,
   cross-time linking.

## Core Data Model (Postgres)

```
patient        (id, name, age, gender, relationship, created_at)
document       (id, patient_id FK, doc_type, classification, file_path,
                source_type, mime_type, raw_ocr_text, status, uploaded_at)
doctor         (id, name, specialty, contact, created_at)
disease        (id, name, icd_code, notes, created_at)
symptom        (id, name, created_at)
medication     (id, name, dosage_form, created_at)
medical_test   (id, name, created_at)
test_result    (id, medical_test_id FK, value, unit, reference_range, observed_at)

-- linking + relationships
document_entity (id, document_id FK, entity_type, entity_id, confidence,
                 validated, source_span)
doctor_prescribed   (id, doctor_id FK, document_id FK, target_type, target_id)
disease_association  (id, disease_id FK, target_type, target_id, document_id FK)

-- RAG
chunk          (id, document_id FK, patient_id, ord, text, page_ref,
                section_ref, embedding vector(768))
```

Notes:
- `patient_id` is duplicated onto `chunk` so cross-patient isolation is a cheap
  `WHERE patient_id = ?` filter on retrieval.
- `document_entity.entity_type` + `entity_id` is a polymorphic link to the six
  entity tables, carrying `confidence` and `validated` for the HITL layer.
- Entity tables are normalized (one row per real-world disease/medication/etc.)
  so duplicate-merge in HITL is a row merge.

---

## Foundation (Sub-project 1) — Detailed Spec

Goal: a runnable skeleton with DB schema and storage in place. No AI, no
ingestion logic yet — just the scaffolding everything else builds on.

### Deliverables

1. **Repo layout**
   ```
   app/
     main.py            # FastAPI app, mounts static UI + API router
     config.py          # settings via env (DB url, Gemini key, storage dir)
     db.py              # engine, session, pgvector setup
     models.py          # SQLAlchemy models for the schema above
     routers/
       health.py        # GET /api/health  -> {status, db, version}
       patients.py      # CRUD for patient profiles (foundation-level, real)
     storage.py         # local file-storage helpers (path layout, save/read)
   migrations/          # Alembic
   static/
     index.html         # placeholder landing page
   tests/
   .env.example
   requirements.txt
   README.md
   ```

2. **Config** (`app/config.py`) — pydantic-settings reading:
   `DATABASE_URL`, `GEMINI_API_KEY`, `STORAGE_DIR` (default `./data/files`),
   `APP_VERSION`.

3. **Database (Supabase)** — `DATABASE_URL` = Supabase Postgres connection
   string (session pooler for the app). Alembic migration creates all tables
   above, including `CREATE EXTENSION IF NOT EXISTS vector` and the
   `chunk.embedding vector(768)` column (768 = `text-embedding-004` dim).
   Supabase MCP tools (`apply_migration`, `list_tables`, `get_advisors`) are
   available and may be used to apply/inspect migrations against the project.

4. **File storage** (`app/storage.py`) — layout
   `STORAGE_DIR/<patient_id>/<document_id>.<ext>`. Functions: `save_upload`,
   `path_for`, `read_file`. Raw bytes never touch the DB.

5. **Patient CRUD** (`routers/patients.py`) — real, since every other
   sub-project needs patients to exist: `POST/GET/GET{id}/PATCH/DELETE`.

6. **Health check** — `GET /api/health` confirms DB connectivity + pgvector
   present; returns version.

7. **Static serving** — FastAPI serves `static/` so the UI sub-project has a
   mount point. Placeholder `index.html`.

### Tech choices (foundation)

- SQLAlchemy 2.x + Alembic for models/migrations.
- `pgvector` Python package for the vector column type.
- `pydantic-settings` for config.
- `pytest` for the test suite (test against a Supabase branch DB or local
  throwaway Postgres).

### Out of scope (foundation)

OCR, Gemini calls, entity extraction, chunking/embeddings, the 5 UI screens,
HITL editing. Those are later sub-projects. Foundation only needs patient CRUD
+ schema + storage + health to be real and tested.

### Success criteria

- `alembic upgrade head` builds the full schema incl. pgvector on a fresh DB.
- `GET /api/health` returns `db: ok` and reports pgvector present.
- Patient CRUD works end-to-end with tests passing.
- `storage.save_upload` writes a file to the patient-scoped path and DB stores
  only the path.
