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
| App | Python + **Streamlit** — single app, UI + logic in one process. |
| Hosting | **Streamlit Community Cloud** (free). |
| AI provider | Google Gemini API only — vision OCR, entity extraction, `text-embedding-004` embeddings, chat (`gemini-2.0-flash`). |
| Database | Supabase (hosted PostgreSQL) + pgvector. Holds structured entities AND embeddings. |
| File storage | **Supabase Storage** bucket. DB stores only a storage reference + metadata. **Only medical-image documents store their raw file for now**; prescriptions/other docs keep OCR text + metadata only (no raw file). |
| Auth | None. Single owner manages multiple patient profiles. Isolation enforced at query level. |
| Architecture | Thin Streamlit view over a **testable service/data layer** (config, db, models, storage, services). Business logic lives in services and is unit-tested without Streamlit. |

> **Note:** Streamlit Community Cloud has an ephemeral filesystem, so raw files can't persist on local disk. Supabase Storage replaces the original "local disk" rule while preserving its intent: the DB holds only references, not file bytes.

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
document       (id, patient_id FK, doc_type, classification, storage_key,
                source_type, mime_type, raw_ocr_text, status, uploaded_at)
                -- storage_key = Supabase Storage object key; NULL for docs
                --   whose raw file we don't keep (non-image, for now)
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
     config.py          # settings via env (DB url, Gemini key, Supabase Storage)
     db.py              # SQLAlchemy engine, session, Base
     models.py          # SQLAlchemy models for the schema above
     storage.py         # Supabase Storage client: upload/get-url/delete (images)
     services/
       patients.py      # patient CRUD functions (used by Streamlit + tests)
       health.py        # db + pgvector connectivity check
   streamlit_app.py     # Streamlit entry: Home — patient list + create form
   migrations/          # Alembic
   tests/
   .env.example
   .streamlit/secrets.toml.example
   requirements.txt
   README.md
   ```

2. **Config** (`app/config.py`) — pydantic-settings reading:
   `DATABASE_URL`, `GEMINI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`,
   `SUPABASE_BUCKET` (default `medical-images`), `APP_VERSION`. On Streamlit
   Cloud these come from `st.secrets`; locally from `.env`.

3. **Database (Supabase)** — `DATABASE_URL` = Supabase Postgres connection
   string (session pooler). Alembic migration creates all tables above,
   including `CREATE EXTENSION IF NOT EXISTS vector` and the
   `chunk.embedding vector(768)` column (768 = `text-embedding-004` dim).
   Supabase MCP tools (`apply_migration`, `list_tables`, `get_advisors`) are
   available and may be used to apply/inspect migrations.

4. **File storage** (`app/storage.py`) — Supabase Storage via `supabase-py`.
   Object key layout `<patient_id>/<document_id>.<ext>` in the
   `SUPABASE_BUCKET` bucket. Functions: `upload_image(patient_id, document_id,
   ext, data, content_type) -> storage_key`, `get_url(storage_key)`,
   `delete(storage_key)`. **Only medical-image documents are uploaded for now**;
   non-image docs store `storage_key = NULL`. Raw bytes never touch the DB.

5. **Patient service** (`app/services/patients.py`) — plain functions, no
   Streamlit/web coupling, every other sub-project depends on them:
   `create_patient`, `list_patients`, `get_patient`, `update_patient`,
   `delete_patient`. Fully unit-tested.

6. **Health check** (`app/services/health.py`) — `check_health() -> dict`
   confirms DB connectivity + pgvector present; returns version.

7. **Streamlit Home** (`streamlit_app.py`) — minimal: lists patients (via
   service) and a "Add patient" form. Thin view; all logic in services. This is
   the mount point the UI sub-project expands into `pages/`.

### Tech choices (foundation)

- Streamlit for the app/UI; logic kept in `app/services/` so it's testable
  headless.
- SQLAlchemy 2.x + Alembic for models/migrations.
- `pgvector` Python package for the vector column type.
- `supabase` (supabase-py) for Storage.
- `pydantic-settings` for config.
- `pytest` for the test suite (DB tests against a Supabase branch DB or local
  throwaway Postgres; Storage upload tested against the Supabase bucket or
  skipped when creds absent).

### Out of scope (foundation)

OCR, Gemini calls, entity extraction, chunking/embeddings, the 5 UI screens,
HITL editing. Those are later sub-projects. Foundation only needs the patient
service + schema + storage client + health to be real and tested, plus a
minimal Streamlit Home.

### Success criteria

- `alembic upgrade head` builds the full schema incl. pgvector on a fresh DB.
- `check_health()` returns `db: ok` and reports pgvector present.
- Patient service CRUD works end-to-end with tests passing.
- `storage.upload_image` puts an image in the Supabase bucket and returns a
  `storage_key`; the DB stores only that key.
- `streamlit run streamlit_app.py` shows the Home page with patient list + form.
