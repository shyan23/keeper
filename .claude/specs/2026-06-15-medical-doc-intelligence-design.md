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
| App | Python **service/data layer** — testable modules, run locally. No UI yet (deferred). |
| Hosting | Local for now. (Deployment + UI revisited later.) |
| AI provider | Google Gemini API only — vision OCR, entity extraction, `text-embedding-004` embeddings, chat (`gemini-2.0-flash`). |
| Database | Supabase (hosted PostgreSQL) + pgvector, reached over the **IPv4 session pooler** (`aws-0-ap-southeast-1.pooler.supabase.com:5432`, user `postgres.<ref>`). Holds structured entities AND embeddings. |
| File storage | **Local disk** under `STORAGE_DIR` (default `./data/files`), patient-scoped layout. DB stores only the file path + metadata; raw bytes never touch the DB. |
| Auth | None. Single owner manages multiple patient profiles. Isolation enforced at query level. |
| Architecture | Plain Python **service/data layer** (config, db, models, storage, services), fully unit-tested. A UI (FastAPI or Streamlit) mounts on top in a later sub-project. |

> **Note:** the DB is Supabase but the app runs locally. Direct Supabase connections are IPv6-only; this machine has no global IPv6, so the **session pooler** (IPv4) URL is used in `DATABASE_URL`.

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
                -- file_path = local disk path under STORAGE_DIR
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
     config.py          # settings via env (DB url, Gemini key, STORAGE_DIR)
     db.py              # SQLAlchemy engine, session, Base
     models.py          # SQLAlchemy models for the schema above
     storage.py         # local-disk file helpers (path layout, save/read)
     services/
       patients.py      # patient CRUD functions (used by tests + later UI)
       health.py        # db + pgvector connectivity check
   migrations/          # Alembic
   tests/
   .env.example
   requirements.txt
   README.md
   ```

2. **Config** (`app/config.py`) — pydantic-settings reading:
   `DATABASE_URL`, `GEMINI_API_KEY`, `STORAGE_DIR` (default `./data/files`),
   `APP_VERSION`. Loaded from `.env` locally.

3. **Database (Supabase via pooler)** — `DATABASE_URL` = Supabase **session
   pooler** connection string (IPv4). Alembic migration creates all tables
   above, including `CREATE EXTENSION IF NOT EXISTS vector` and the
   `chunk.embedding vector(768)` column (768 = `text-embedding-004` dim).
   Supabase MCP tools (`apply_migration`, `list_tables`, `get_advisors`) are
   available and may be used to apply/inspect migrations.

4. **File storage** (`app/storage.py`) — local disk, layout
   `STORAGE_DIR/<patient_id>/<document_id>.<ext>`. Functions:
   `save_bytes(patient_id, document_id, ext, data) -> path`,
   `path_for(patient_id, document_id, ext)`, `read_file(path)`. Raw bytes never
   touch the DB; the DB stores only the path.

5. **Patient service** (`app/services/patients.py`) — plain functions, no UI
   coupling, every other sub-project depends on them: `create_patient`,
   `list_patients`, `get_patient`, `update_patient`, `delete_patient`. Fully
   unit-tested.

6. **Health check** (`app/services/health.py`) — `check_health() -> dict`
   confirms DB connectivity + pgvector present; returns version.

### Tech choices (foundation)

- Plain Python `app/services/` layer; fully testable headless. UI deferred.
- SQLAlchemy 2.x + Alembic for models/migrations.
- `pgvector` Python package for the vector column type.
- `pydantic-settings` for config.
- `pytest` for the test suite (DB tests against the Supabase pooler DB).

### Out of scope (foundation)

OCR, Gemini calls, entity extraction, chunking/embeddings, any UI, HITL editing.
Those are later sub-projects. Foundation only needs the patient service + schema
+ local storage helpers + health to be real and tested.

### Success criteria

- `alembic upgrade head` builds the full schema incl. pgvector on a fresh DB.
- `check_health()` returns `db: ok` and reports pgvector present.
- Patient service CRUD works end-to-end with tests passing.
- `storage.save_bytes` writes a file to the patient-scoped path and returns it;
  the DB stores only the path.
