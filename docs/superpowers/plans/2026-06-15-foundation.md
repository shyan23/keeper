# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Streamlit + Supabase(Postgres/pgvector) skeleton with the full DB schema, a Supabase Storage client for medical images, a testable patient service, a health check, and a minimal Streamlit Home — the scaffold every later sub-project builds on.

**Architecture:** A thin Streamlit view over a testable service/data layer. SQLAlchemy 2.x models + Alembic migrations against Supabase Postgres with the `vector` extension. Raw files (medical images only, for now) go to a Supabase Storage bucket; the DB stores only a `storage_key` reference + metadata. All business logic lives in `app/services/` and is unit-tested without Streamlit.

**Tech Stack:** Python 3.11+, Streamlit, SQLAlchemy 2.x, Alembic, psycopg (v3), pgvector, supabase-py, pydantic-settings, pytest.

---

## File Structure

```
app/
  __init__.py
  config.py             # pydantic-settings: DATABASE_URL, GEMINI_API_KEY, SUPABASE_URL/KEY/BUCKET, APP_VERSION
  db.py                 # SQLAlchemy engine + session factory + Base + session ctx
  models.py             # all ORM models (patient, document, entities, links, chunk)
  storage.py            # Supabase Storage client: upload_image / get_url / delete
  services/
    __init__.py
    patients.py         # patient CRUD functions
    health.py           # check_health()
streamlit_app.py        # Streamlit entry: Home (patient list + add form)
migrations/
  env.py
  versions/0001_initial.py
tests/
  __init__.py
  conftest.py           # schema setup + per-test cleanup fixtures
  test_config.py
  test_db.py
  test_models.py
  test_migration.py
  test_storage.py
  test_patients_service.py
  test_health_service.py
  test_streamlit_import.py
alembic.ini
.env.example
.streamlit/secrets.toml.example
requirements.txt
README.md
```

**Test DB:** DB-touching tests run against `TEST_DATABASE_URL` (Supabase branch DB or throwaway local Postgres). `conftest.py` ensures the `vector` extension + tables exist and cleans rows per test. No mocking of Postgres.

**Storage tests:** `test_storage.py` hits the real Supabase bucket; it **skips** when `SUPABASE_URL`/`SUPABASE_KEY` are absent so the suite still runs offline.

---

### Task 1: Scaffold + dependencies + config

**Files:**
- Create: `requirements.txt`, `.env.example`, `.streamlit/secrets.toml.example`
- Create: `app/__init__.py`, `app/config.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
streamlit==1.41.1
sqlalchemy==2.0.36
alembic==1.14.0
psycopg[binary]==3.2.3
pgvector==0.3.6
supabase==2.10.0
pydantic-settings==2.7.0
pytest==8.3.4
```

- [ ] **Step 2: Write `.env.example`**

```
DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
TEST_DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
GEMINI_API_KEY=changeme
SUPABASE_URL=https://[PROJECT].supabase.co
SUPABASE_KEY=changeme
SUPABASE_BUCKET=medical-images
APP_VERSION=0.1.0
```

- [ ] **Step 3: Write `.streamlit/secrets.toml.example`**

```toml
DATABASE_URL = "postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres"
GEMINI_API_KEY = "changeme"
SUPABASE_URL = "https://[PROJECT].supabase.co"
SUPABASE_KEY = "changeme"
SUPABASE_BUCKET = "medical-images"
APP_VERSION = "0.1.0"
```

- [ ] **Step 4: Create empty `app/__init__.py` and `tests/__init__.py`**

- [ ] **Step 5: Write the failing test** — `tests/test_config.py`

```python
from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.setenv("SUPABASE_URL", "https://x.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "sk")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/db"
    assert s.gemini_api_key == "key123"
    assert s.supabase_url == "https://x.supabase.co"
    assert s.supabase_key == "sk"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.delenv("SUPABASE_BUCKET", raising=False)
    s = Settings()
    assert s.supabase_bucket == "medical-images"
    assert s.app_version == "0.1.0"
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pip install -r requirements.txt && pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 7: Implement `app/config.py`**

```python
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    test_database_url: str | None = None
    gemini_api_key: str = "changeme"
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_bucket: str = "medical-images"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Commit**

```bash
git add requirements.txt .env.example .streamlit/ app/__init__.py app/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: scaffold, dependencies, and settings"
```

---

### Task 2: Database engine + session

**Files:**
- Create: `app/db.py`
- Test: `tests/test_db.py`

- [ ] **Step 1: Write the failing test** — `tests/test_db.py`

```python
from sqlalchemy import text
from app.db import engine, SessionLocal, Base


def test_engine_and_session_run_sql():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
    db = SessionLocal()
    try:
        assert db.execute(text("SELECT 1")).scalar() == 1
    finally:
        db.close()


def test_base_metadata_present():
    assert Base.metadata is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.db'`

- [ ] **Step 3: Implement `app/db.py`**

```python
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()
_db_url = settings.test_database_url or settings.database_url

engine = create_engine(_db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def session_scope() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_db.py -v`
Expected: PASS (2 passed). Requires `DATABASE_URL` exported to a reachable Supabase/Postgres.

- [ ] **Step 5: Commit**

```bash
git add app/db.py tests/test_db.py
git commit -m "feat: SQLAlchemy engine, session factory, Base"
```

---

### Task 3: ORM models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test** — `tests/test_models.py`

```python
from app.models import (
    Patient, Document, Doctor, Disease, Symptom, Medication,
    MedicalTest, TestResult, DocumentEntity, Chunk,
)
from app.db import Base


def test_tables_registered():
    expected = {
        "patient", "document", "doctor", "disease", "symptom",
        "medication", "medical_test", "test_result",
        "document_entity", "chunk",
    }
    assert expected.issubset(set(Base.metadata.tables.keys()))


def test_document_has_storage_key():
    cols = {c.name for c in Document.__table__.columns}
    assert "storage_key" in cols
    assert "file_path" not in cols
    assert "raw_ocr_text" in cols


def test_chunk_has_vector_and_patient():
    cols = {c.name for c in Chunk.__table__.columns}
    assert {"embedding", "patient_id", "document_id"} <= cols


def test_patient_columns():
    cols = {c.name for c in Patient.__table__.columns}
    assert {"id", "name", "age", "gender", "relationship", "created_at"} <= cols
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Implement `app/models.py`**

```python
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base

EMBED_DIM = 768  # text-embedding-004


class Patient(Base):
    __tablename__ = "patient"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relationship: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    documents: Mapped[list["Document"]] = relationship(back_populates="patient", cascade="all, delete-orphan")


class Document(Base):
    __tablename__ = "document"
    id: Mapped[int] = mapped_column(primary_key=True)
    patient_id: Mapped[int] = mapped_column(ForeignKey("patient.id", ondelete="CASCADE"), index=True)
    doc_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    classification: Mapped[str | None] = mapped_column(String(50), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(Text, nullable=True)  # Supabase Storage object key; NULL when raw file not kept
    source_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    mime_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    raw_ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(30), default="uploaded")
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    patient: Mapped["Patient"] = relationship(back_populates="documents")


class Doctor(Base):
    __tablename__ = "doctor"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    specialty: Mapped[str | None] = mapped_column(String(120), nullable=True)
    contact: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Disease(Base):
    __tablename__ = "disease"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    icd_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Symptom(Base):
    __tablename__ = "symptom"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Medication(Base):
    __tablename__ = "medication"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    dosage_form: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class MedicalTest(Base):
    __tablename__ = "medical_test"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TestResult(Base):
    __tablename__ = "test_result"
    id: Mapped[int] = mapped_column(primary_key=True)
    medical_test_id: Mapped[int] = mapped_column(ForeignKey("medical_test.id", ondelete="CASCADE"), index=True)
    value: Mapped[str | None] = mapped_column(String(120), nullable=True)
    unit: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reference_range: Mapped[str | None] = mapped_column(String(120), nullable=True)
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentEntity(Base):
    __tablename__ = "document_entity"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    entity_type: Mapped[str] = mapped_column(String(30))  # doctor|disease|symptom|medication|medical_test|test_result
    entity_id: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    source_span: Mapped[str | None] = mapped_column(Text, nullable=True)


class Chunk(Base):
    __tablename__ = "chunk"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("document.id", ondelete="CASCADE"), index=True)
    patient_id: Mapped[int] = mapped_column(Integer, index=True)
    ord: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    page_ref: Mapped[str | None] = mapped_column(String(40), nullable=True)
    section_ref: Mapped[str | None] = mapped_column(String(120), nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIM), nullable=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: ORM models for patients, documents, entities, links, chunks"
```

---

### Task 4: Alembic migration (schema + pgvector)

**Files:**
- Create: `alembic.ini`, `migrations/env.py`, `migrations/versions/0001_initial.py`
- Test: `tests/test_migration.py`

- [ ] **Step 1: Create `alembic.ini`**

```ini
[alembic]
script_location = migrations
sqlalchemy.url =

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 2: Create `migrations/env.py`**

```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from app.config import get_settings
from app.db import Base
import app.models  # noqa: F401  (register tables on Base.metadata)

config = context.config
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.test_database_url or settings.database_url)
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
```

- [ ] **Step 3: Create `migrations/versions/0001_initial.py`**

```python
"""initial schema + pgvector

Revision ID: 0001
Revises:
"""
import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "patient",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("age", sa.Integer),
        sa.Column("gender", sa.String(20)),
        sa.Column("relationship", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "document",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("patient_id", sa.Integer, sa.ForeignKey("patient.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("doc_type", sa.String(50)),
        sa.Column("classification", sa.String(50)),
        sa.Column("storage_key", sa.Text),
        sa.Column("source_type", sa.String(20)),
        sa.Column("mime_type", sa.String(100)),
        sa.Column("raw_ocr_text", sa.Text),
        sa.Column("status", sa.String(30), server_default="uploaded"),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "doctor",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("specialty", sa.String(120)),
        sa.Column("contact", sa.String(200)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "disease",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("icd_code", sa.String(20)),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "symptom",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "medication",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("dosage_form", sa.String(80)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "medical_test",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        "test_result",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("medical_test_id", sa.Integer, sa.ForeignKey("medical_test.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("value", sa.String(120)),
        sa.Column("unit", sa.String(40)),
        sa.Column("reference_range", sa.String(120)),
        sa.Column("observed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "document_entity",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("entity_type", sa.String(30), nullable=False),
        sa.Column("entity_id", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("validated", sa.Boolean, server_default=sa.false()),
        sa.Column("source_span", sa.Text),
    )
    op.create_table(
        "chunk",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("document.id", ondelete="CASCADE"), index=True, nullable=False),
        sa.Column("patient_id", sa.Integer, index=True, nullable=False),
        sa.Column("ord", sa.Integer, server_default="0"),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("page_ref", sa.String(40)),
        sa.Column("section_ref", sa.String(120)),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(768)),
    )


def downgrade() -> None:
    for t in ["chunk", "document_entity", "test_result", "medical_test",
              "medication", "symptom", "disease", "doctor", "document", "patient"]:
        op.drop_table(t)
```

- [ ] **Step 4: Write the failing test** — `tests/test_migration.py`

```python
import subprocess

from sqlalchemy import text
from app.db import engine


def _run(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


def test_migration_builds_schema_and_vector():
    _run("alembic downgrade base || true")
    _run("alembic upgrade head")
    with engine.connect() as conn:
        ext = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar()
        assert ext == 1
        tables = conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )).scalars().all()
        assert "patient" in tables
        assert "chunk" in tables
        cols = conn.execute(text(
            "SELECT column_name FROM information_schema.columns WHERE table_name='document'"
        )).scalars().all()
        assert "storage_key" in cols
```

- [ ] **Step 5: Run test to verify it passes**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_migration.py -v`
Expected: PASS once the migration files exist (before then, alembic errors → FAIL).

- [ ] **Step 6: Commit**

```bash
git add alembic.ini migrations/ tests/test_migration.py
git commit -m "feat: alembic initial migration with full schema and pgvector"
```

---

### Task 5: Supabase Storage client (medical images)

**Files:**
- Create: `app/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test** — `tests/test_storage.py`

```python
import os
import uuid

import pytest

from app.storage import upload_image, get_url, delete, object_key

HAS_CREDS = bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY"))


def test_object_key_layout():
    assert object_key(7, 42, "png") == "7/42.png"
    assert object_key(1, 2, ".jpg") == "1/2.jpg"


@pytest.mark.skipif(not HAS_CREDS, reason="Supabase creds not set")
def test_upload_get_delete_roundtrip():
    pid = 999000 + (uuid.uuid4().int % 1000)
    key = upload_image(patient_id=pid, document_id=1, ext="png",
                       data=b"\x89PNG\r\n\x1a\n fake", content_type="image/png")
    assert key == f"{pid}/1.png"
    url = get_url(key)
    assert isinstance(url, str) and url.startswith("http")
    delete(key)  # cleanup; should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage'`

- [ ] **Step 3: Implement `app/storage.py`**

```python
from functools import lru_cache

from supabase import Client, create_client

from app.config import get_settings


@lru_cache
def _client() -> Client:
    s = get_settings()
    if not s.supabase_url or not s.supabase_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not configured")
    return create_client(s.supabase_url, s.supabase_key)


def object_key(patient_id: int, document_id: int, ext: str) -> str:
    return f"{patient_id}/{document_id}.{ext.lstrip('.')}"


def upload_image(patient_id: int, document_id: int, ext: str, data: bytes,
                 content_type: str = "application/octet-stream") -> str:
    """Upload a medical image to Supabase Storage; return the object key."""
    key = object_key(patient_id, document_id, ext)
    bucket = get_settings().supabase_bucket
    _client().storage.from_(bucket).upload(
        path=key, file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return key


def get_url(storage_key: str) -> str:
    bucket = get_settings().supabase_bucket
    return _client().storage.from_(bucket).get_public_url(storage_key)


def delete(storage_key: str) -> None:
    bucket = get_settings().supabase_bucket
    _client().storage.from_(bucket).remove([storage_key])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS — `test_object_key_layout` passes; roundtrip passes if Supabase creds + a `medical-images` bucket exist, else SKIPPED.

- [ ] **Step 5: Commit**

```bash
git add app/storage.py tests/test_storage.py
git commit -m "feat: Supabase Storage client for medical images"
```

---

### Task 6: Patient service + health service

**Files:**
- Create: `app/services/__init__.py`, `app/services/patients.py`, `app/services/health.py`
- Test: `tests/conftest.py`, `tests/test_patients_service.py`, `tests/test_health_service.py`

- [ ] **Step 1: Create `tests/conftest.py`**

```python
import os

import pytest

os.environ.setdefault("TEST_DATABASE_URL", os.environ.get("DATABASE_URL", ""))

from app.db import Base, engine, SessionLocal  # noqa: E402
import app.models  # noqa: E402,F401


@pytest.fixture(scope="session", autouse=True)
def _schema():
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(engine)
    yield


@pytest.fixture(autouse=True)
def _clean_patients():
    db = SessionLocal()
    try:
        from app.models import Patient
        db.query(Patient).delete()
        db.commit()
    finally:
        db.close()
    yield


@pytest.fixture
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()
```

- [ ] **Step 2: Create `app/services/__init__.py`** (empty file)

- [ ] **Step 3: Write the failing test** — `tests/test_patients_service.py`

```python
import pytest

from app.services import patients as svc


def test_create_and_get(db):
    p = svc.create_patient(db, name="Asha", age=30, relationship="mother")
    assert p.id is not None
    fetched = svc.get_patient(db, p.id)
    assert fetched.name == "Asha"
    assert fetched.relationship == "mother"


def test_list_orders_by_id(db):
    a = svc.create_patient(db, name="A")
    b = svc.create_patient(db, name="B")
    ids = [p.id for p in svc.list_patients(db)]
    assert ids == sorted(ids)
    assert a.id in ids and b.id in ids


def test_update(db):
    p = svc.create_patient(db, name="Asha", age=30)
    updated = svc.update_patient(db, p.id, age=31)
    assert updated.age == 31


def test_delete(db):
    p = svc.create_patient(db, name="Asha")
    svc.delete_patient(db, p.id)
    assert svc.get_patient(db, p.id) is None


def test_get_missing_returns_none(db):
    assert svc.get_patient(db, 999999) is None


def test_update_missing_raises(db):
    with pytest.raises(ValueError):
        svc.update_patient(db, 999999, age=1)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_patients_service.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.services.patients'`

- [ ] **Step 5: Implement `app/services/patients.py`**

```python
from sqlalchemy.orm import Session

from app.models import Patient


def create_patient(db: Session, *, name: str, age: int | None = None,
                   gender: str | None = None, relationship: str | None = None) -> Patient:
    patient = Patient(name=name, age=age, gender=gender, relationship=relationship)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def list_patients(db: Session) -> list[Patient]:
    return db.query(Patient).order_by(Patient.id).all()


def get_patient(db: Session, patient_id: int) -> Patient | None:
    return db.get(Patient, patient_id)


def update_patient(db: Session, patient_id: int, **fields) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"patient {patient_id} not found")
    for key, value in fields.items():
        if value is not None:
            setattr(patient, key, value)
    db.commit()
    db.refresh(patient)
    return patient


def delete_patient(db: Session, patient_id: int) -> None:
    patient = db.get(Patient, patient_id)
    if patient is not None:
        db.delete(patient)
        db.commit()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_patients_service.py -v`
Expected: PASS (6 passed)

- [ ] **Step 7: Write the failing test** — `tests/test_health_service.py`

```python
from app.services.health import check_health


def test_check_health_ok():
    result = check_health()
    assert result["db"] == "ok"
    assert result["pgvector"] is True
    assert "version" in result
```

- [ ] **Step 8: Implement `app/services/health.py`**

```python
from sqlalchemy import text

from app.config import get_settings
from app.db import SessionLocal


def check_health() -> dict:
    db = SessionLocal()
    db_ok = False
    vector_ok = False
    try:
        db_ok = db.execute(text("SELECT 1")).scalar() == 1
        vector_ok = bool(db.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar())
    except Exception:
        db_ok = False
    finally:
        db.close()
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "pgvector": vector_ok,
        "version": get_settings().app_version,
    }
```

- [ ] **Step 9: Run test to verify it passes**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_health_service.py -v`
Expected: PASS (1 passed). Requires the `vector` extension present (conftest creates it).

- [ ] **Step 10: Commit**

```bash
git add app/services/ tests/conftest.py tests/test_patients_service.py tests/test_health_service.py
git commit -m "feat: patient service + health service with tests"
```

---

### Task 7: Streamlit Home

**Files:**
- Create: `streamlit_app.py`
- Test: `tests/test_streamlit_import.py`

- [ ] **Step 1: Write the failing test** — `tests/test_streamlit_import.py`

```python
import importlib


def test_streamlit_app_exposes_main():
    mod = importlib.import_module("streamlit_app")
    assert hasattr(mod, "main")
    assert callable(mod.main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streamlit_import.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'streamlit_app'`

- [ ] **Step 3: Implement `streamlit_app.py`**

```python
import streamlit as st

from app.db import SessionLocal
from app.services import patients as patient_svc
from app.services.health import check_health


def main() -> None:
    st.set_page_config(page_title="Medical Document Intelligence", layout="wide")
    st.title("Medical Document Intelligence")

    health = check_health()
    if health["db"] == "ok" and health["pgvector"]:
        st.caption(f"DB ok · pgvector ready · v{health['version']}")
    else:
        st.error(f"Backend not ready: {health}")

    db = SessionLocal()
    try:
        st.subheader("Patients")
        people = patient_svc.list_patients(db)
        if people:
            st.table([
                {"id": p.id, "name": p.name, "age": p.age,
                 "gender": p.gender, "relationship": p.relationship}
                for p in people
            ])
        else:
            st.info("No patients yet. Add one below.")

        with st.form("add_patient", clear_on_submit=True):
            st.write("Add patient")
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=0, max_value=130, value=0, step=1)
            gender = st.selectbox("Gender", ["", "male", "female", "other"])
            relationship = st.text_input("Relationship (e.g. mother, self)")
            submitted = st.form_submit_button("Save")
            if submitted and name.strip():
                patient_svc.create_patient(
                    db, name=name.strip(),
                    age=int(age) or None,
                    gender=gender or None,
                    relationship=relationship.strip() or None,
                )
                st.success(f"Added {name}")
                st.rerun()
    finally:
        db.close()


main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streamlit_import.py -v`
Expected: PASS. (Importing runs Streamlit calls outside a script-run context, which only logs "missing ScriptRunContext" warnings — not errors. The `main` attribute exists.)

- [ ] **Step 5: Manual smoke check**

Run: `TEST_DATABASE_URL=$DATABASE_URL streamlit run streamlit_app.py`
Expected: Home page loads, shows "DB ok · pgvector ready", patient table/empty state, and an Add-patient form that persists a row on submit.

- [ ] **Step 6: Commit**

```bash
git add streamlit_app.py tests/test_streamlit_import.py
git commit -m "feat: Streamlit Home with patient list and add form"
```

---

### Task 8: README + full suite green

**Files:**
- Create: `README.md`
- Test: full `pytest` run

- [ ] **Step 1: Create `README.md`**

```markdown
# Medical Document Intelligence & Tracker

Foundation sub-project: Streamlit + Supabase(Postgres/pgvector) skeleton.

## Setup (local)

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`; set `DATABASE_URL` (Supabase Postgres),
   `SUPABASE_URL`, `SUPABASE_KEY`, `SUPABASE_BUCKET`, `GEMINI_API_KEY`.
4. In Supabase, create a Storage bucket named `medical-images`.
5. `alembic upgrade head`
6. `streamlit run streamlit_app.py`

## Deploy (free)

Push to GitHub, then deploy on Streamlit Community Cloud. Put the same keys in
the app's Secrets (see `.streamlit/secrets.toml.example`). Supabase holds the
database and file storage, so data persists across app restarts.

## Tests

`TEST_DATABASE_URL=<postgres-url> pytest -v`

DB tests run against a real Postgres (Supabase branch DB or throwaway local) so
pgvector behaviour is exercised, not mocked. The Supabase Storage roundtrip test
is skipped unless `SUPABASE_URL` / `SUPABASE_KEY` are set.

## Build order

See `docs/superpowers/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
```

- [ ] **Step 2: Run the full suite**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest -v`
Expected: all tests pass (Storage roundtrip skipped if no creds).

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup, deploy, and test instructions"
```

---

## Self-Review Notes

- **Spec coverage:** scaffold + config incl. Supabase vars (T1), DB engine (T2), full schema models with `storage_key` (T3), Alembic migration + pgvector + `vector(768)` (T4), Supabase Storage client for images only (T5), patient service + health service (T6), Streamlit Home thin view (T7), README/deploy/success criteria (T8). All Foundation deliverables and success criteria mapped.
- **Streamlit Cloud / persistence:** files go to Supabase Storage, DB to Supabase Postgres — nothing relies on local disk, so the ephemeral filesystem is a non-issue. Only medical images are uploaded for now (`storage_key` NULL otherwise), per decision.
- **Type consistency:** `Settings` fields; `engine`/`SessionLocal`/`Base` (db.py); model class names incl. `Document.storage_key`; storage funcs `object_key`/`upload_image`/`get_url`/`delete`; service funcs `create_patient`/`list_patients`/`get_patient`/`update_patient`/`delete_patient`; `check_health`; `streamlit_app.main` — all referenced consistently across tasks.
- **Out of scope (later sub-projects):** OCR, Gemini calls, entity extraction, chunking/embeddings population, the 5 UI screens, HITL editing.
- **Known constraint:** DB-touching tests need a reachable Postgres with permission to `CREATE EXTENSION vector` (Supabase grants this); Storage test needs a `medical-images` bucket or it skips.
