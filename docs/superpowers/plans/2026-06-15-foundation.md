# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable FastAPI + Supabase(Postgres/pgvector) skeleton with the full DB schema, local file storage, patient CRUD, and a health check — the scaffold every later sub-project builds on.

**Architecture:** One FastAPI monolith serving an API router plus static files. SQLAlchemy 2.x models + Alembic migrations against a Supabase Postgres database with the `vector` extension. Raw files live on local disk under a patient-scoped layout; the DB stores only paths + metadata.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, SQLAlchemy 2.x, Alembic, psycopg (v3), pgvector, pydantic-settings, pytest, httpx (TestClient).

---

## File Structure

```
app/
  __init__.py
  config.py            # pydantic-settings: DATABASE_URL, GEMINI_API_KEY, STORAGE_DIR, APP_VERSION
  db.py                # SQLAlchemy engine + session factory + get_db dependency
  models.py            # all ORM models (patient, document, entities, links, chunk)
  storage.py           # local file storage helpers
  schemas.py           # pydantic request/response models (patient)
  main.py              # FastAPI app, router includes, static mount
  routers/
    __init__.py
    health.py          # GET /api/health
    patients.py        # patient CRUD
migrations/            # Alembic env + versions
  env.py
  versions/0001_initial.py
static/
  index.html
tests/
  __init__.py
  conftest.py          # test DB session + TestClient fixtures
  test_health.py
  test_storage.py
  test_patients.py
alembic.ini
.env.example
requirements.txt
README.md
```

**Note on test DB:** tests run against the Postgres pointed to by `TEST_DATABASE_URL` (a Supabase branch DB or a throwaway local Postgres). `conftest.py` creates tables via `Base.metadata.create_all` against that DB and rolls back per test. No mocking of Postgres — pgvector behavior must be real.

---

### Task 1: Project scaffold + dependencies + config

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
alembic==1.14.0
psycopg[binary]==3.2.3
pgvector==0.3.6
pydantic-settings==2.7.0
python-multipart==0.0.20
httpx==0.28.1
pytest==8.3.4
```

- [ ] **Step 2: Create `.env.example`**

```
DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
TEST_DATABASE_URL=postgresql+psycopg://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres
GEMINI_API_KEY=changeme
STORAGE_DIR=./data/files
APP_VERSION=0.1.0
```

- [ ] **Step 3: Create empty `app/__init__.py` and `tests/__init__.py`**

Both empty files.

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
import os
from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.setenv("STORAGE_DIR", "/tmp/files")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/db"
    assert s.gemini_api_key == "key123"
    assert s.storage_dir == "/tmp/files"


def test_settings_storage_dir_default(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    s = Settings()
    assert s.storage_dir == "./data/files"
    assert s.app_version == "0.1.0"
```

- [ ] **Step 5: Run test to verify it fails**

Run: `pip install -r requirements.txt && pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.config'`

- [ ] **Step 6: Implement `app/config.py`**

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    test_database_url: str | None = None
    gemini_api_key: str = "changeme"
    storage_dir: str = "./data/files"
    app_version: str = "0.1.0"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 7: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example app/__init__.py app/config.py tests/__init__.py tests/test_config.py
git commit -m "feat: project scaffold, dependencies, and settings"
```

---

### Task 2: Database engine + session

**Files:**
- Create: `app/db.py`
- Test: covered indirectly via later fixtures; add a smoke test `tests/test_db.py`

- [ ] **Step 1: Write the failing test** — `tests/test_db.py`

```python
from sqlalchemy import text
from app.db import engine, SessionLocal, Base


def test_engine_connects_and_session_works():
    with engine.connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar() == 1
    db = SessionLocal()
    try:
        assert db.execute(text("SELECT 1")).scalar() == 1
    finally:
        db.close()


def test_base_has_metadata():
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
# Tests set TEST_DATABASE_URL; app uses DATABASE_URL.
_db_url = settings.test_database_url or settings.database_url

engine = create_engine(_db_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
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
git commit -m "feat: SQLAlchemy engine, session factory, Base, get_db dependency"
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


def test_tables_registered():
    expected = {
        "patient", "document", "doctor", "disease", "symptom",
        "medication", "medical_test", "test_result",
        "document_entity", "chunk",
    }
    from app.db import Base
    assert expected.issubset(set(Base.metadata.tables.keys()))


def test_chunk_has_vector_embedding():
    cols = {c.name: c for c in Chunk.__table__.columns}
    assert "embedding" in cols
    assert "patient_id" in cols
    assert "document_id" in cols


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
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, Boolean, func
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
    file_path: Mapped[str] = mapped_column(Text)
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
    confidence: Mapped[float | None] = mapped_column(nullable=True)
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
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: ORM models for patients, documents, entities, links, chunks"
```

---

### Task 4: Alembic migration (schema + pgvector)

**Files:**
- Create: `alembic.ini`
- Create: `migrations/env.py`
- Create: `migrations/versions/0001_initial.py`
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
        sa.Column("file_path", sa.Text, nullable=False),
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
```

- [ ] **Step 5: Run test to verify it fails then passes**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_migration.py -v`
Expected: After creating the files above, PASS. (Before `migrations/` exists, alembic errors → FAIL.)

- [ ] **Step 6: Commit**

```bash
git add alembic.ini migrations/ tests/test_migration.py
git commit -m "feat: alembic initial migration with full schema and pgvector"
```

---

### Task 5: Local file storage helpers

**Files:**
- Create: `app/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test** — `tests/test_storage.py`

```python
from pathlib import Path

from app.storage import save_bytes, path_for, read_file


def test_save_and_read(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    import app.storage as storage
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))

    p = save_bytes(patient_id=7, document_id=42, ext="pdf", data=b"%PDF-1.4 hi")
    assert Path(p).exists()
    assert p == str(tmp_path / "7" / "42.pdf")
    assert read_file(p) == b"%PDF-1.4 hi"


def test_path_for(monkeypatch, tmp_path):
    import app.storage as storage
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    assert path_for(1, 2, "png") == str(tmp_path / "1" / "2.png")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage'`

- [ ] **Step 3: Implement `app/storage.py`**

```python
import os
from pathlib import Path

from app.config import get_settings

STORAGE_DIR = get_settings().storage_dir


def path_for(patient_id: int, document_id: int, ext: str) -> str:
    ext = ext.lstrip(".")
    return str(Path(STORAGE_DIR) / str(patient_id) / f"{document_id}.{ext}")


def save_bytes(patient_id: int, document_id: int, ext: str, data: bytes) -> str:
    target = Path(path_for(patient_id, document_id, ext))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)


def read_file(path: str) -> bytes:
    return Path(path).read_bytes()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_storage.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/storage.py tests/test_storage.py
git commit -m "feat: local file storage helpers with patient-scoped layout"
```

---

### Task 6: Pydantic schemas + patient router + health router + app wiring

**Files:**
- Create: `app/schemas.py`
- Create: `app/routers/__init__.py`
- Create: `app/routers/health.py`
- Create: `app/routers/patients.py`
- Create: `app/main.py`
- Create: `static/index.html`
- Test: `tests/conftest.py`, `tests/test_health.py`, `tests/test_patients.py`

- [ ] **Step 1: Create `app/schemas.py`**

```python
from pydantic import BaseModel, ConfigDict


class PatientCreate(BaseModel):
    name: str
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None


class PatientUpdate(BaseModel):
    name: str | None = None
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None


class PatientOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None
```

- [ ] **Step 2: Create `app/routers/__init__.py`** (empty file)

- [ ] **Step 3: Create `app/routers/health.py`**

```python
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import get_db

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db_ok = False
    vector_ok = False
    try:
        db_ok = db.execute(text("SELECT 1")).scalar() == 1
        vector_ok = bool(db.execute(text("SELECT 1 FROM pg_extension WHERE extname='vector'")).scalar())
    except Exception:
        db_ok = False
    return {
        "status": "ok" if db_ok else "degraded",
        "db": "ok" if db_ok else "error",
        "pgvector": vector_ok,
        "version": get_settings().app_version,
    }
```

- [ ] **Step 4: Create `app/routers/patients.py`**

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Patient
from app.schemas import PatientCreate, PatientOut, PatientUpdate

router = APIRouter(prefix="/api/patients", tags=["patients"])


@router.post("", response_model=PatientOut, status_code=201)
def create_patient(payload: PatientCreate, db: Session = Depends(get_db)):
    patient = Patient(**payload.model_dump())
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


@router.get("", response_model=list[PatientOut])
def list_patients(db: Session = Depends(get_db)):
    return db.query(Patient).order_by(Patient.id).all()


@router.get("/{patient_id}", response_model=PatientOut)
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")
    return patient


@router.patch("/{patient_id}", response_model=PatientOut)
def update_patient(patient_id: int, payload: PatientUpdate, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")
    for k, v in payload.model_dump(exclude_unset=True).items():
        setattr(patient, k, v)
    db.commit()
    db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=204)
def delete_patient(patient_id: int, db: Session = Depends(get_db)):
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="patient not found")
    db.delete(patient)
    db.commit()
```

- [ ] **Step 5: Create `static/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Medical Document Intelligence</title>
</head>
<body>
  <h1>Medical Document Intelligence</h1>
  <p>Foundation running. UI sub-project mounts here.</p>
</body>
</html>
```

- [ ] **Step 6: Create `app/main.py`**

```python
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routers import health, patients

app = FastAPI(title="Medical Document Intelligence", version=get_settings().app_version)
app.include_router(health.router)
app.include_router(patients.router)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 7: Create `tests/conftest.py`**

```python
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("TEST_DATABASE_URL", os.environ.get("DATABASE_URL", ""))

from app.db import Base, engine, SessionLocal  # noqa: E402
import app.models  # noqa: E402,F401
from app.main import app  # noqa: E402


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
def client():
    return TestClient(app)
```

- [ ] **Step 8: Write the failing tests** — `tests/test_health.py`

```python
def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] == "ok"
    assert body["pgvector"] is True
    assert "version" in body
```

and `tests/test_patients.py`

```python
def test_patient_crud(client):
    r = client.post("/api/patients", json={"name": "Asha", "age": 30, "relationship": "mother"})
    assert r.status_code == 201
    pid = r.json()["id"]
    assert r.json()["name"] == "Asha"

    r = client.get("/api/patients")
    assert r.status_code == 200
    assert any(p["id"] == pid for p in r.json())

    r = client.get(f"/api/patients/{pid}")
    assert r.status_code == 200

    r = client.patch(f"/api/patients/{pid}", json={"age": 31})
    assert r.status_code == 200
    assert r.json()["age"] == 31

    r = client.delete(f"/api/patients/{pid}")
    assert r.status_code == 204

    r = client.get(f"/api/patients/{pid}")
    assert r.status_code == 404


def test_get_missing_patient_404(client):
    r = client.get("/api/patients/999999")
    assert r.status_code == 404
```

- [ ] **Step 9: Run tests to verify they fail then pass**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest tests/test_health.py tests/test_patients.py -v`
Expected: PASS once all files above exist (health: 1 passed, patients: 2 passed).

- [ ] **Step 10: Commit**

```bash
git add app/schemas.py app/routers/ app/main.py static/index.html tests/conftest.py tests/test_health.py tests/test_patients.py
git commit -m "feat: health + patient CRUD routers, app wiring, static mount"
```

---

### Task 7: README + full suite green

**Files:**
- Create: `README.md`
- Test: full `pytest` run

- [ ] **Step 1: Create `README.md`**

```markdown
# Medical Document Intelligence & Tracker

Foundation sub-project: FastAPI + Supabase(Postgres/pgvector) skeleton.

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, set `DATABASE_URL` to your Supabase Postgres
   connection string and `GEMINI_API_KEY`.
4. `alembic upgrade head`
5. `uvicorn app.main:app --reload`

Open http://localhost:8000/ (static UI) and http://localhost:8000/api/health.

## Tests

`TEST_DATABASE_URL=<postgres-url> pytest -v`

Tests run against a real Postgres (Supabase branch DB or throwaway local) so
pgvector behaviour is exercised, not mocked.

## Build order

See `docs/superpowers/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
```

- [ ] **Step 2: Run the full suite**

Run: `TEST_DATABASE_URL=$DATABASE_URL pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and test instructions"
```

---

## Self-Review Notes

- **Spec coverage:** repo scaffold (T1), config (T1), DB engine (T2), full schema models (T3), Alembic migration + pgvector + `vector(768)` (T4), local patient-scoped file storage (T5), patient CRUD + health + static mount (T6), README/success criteria (T7). All Foundation deliverables and success criteria mapped.
- **Out of scope (deferred to later sub-projects):** OCR, Gemini calls, entity extraction, chunking/embeddings population, the 5 UI screens, HITL editing. Correct per spec.
- **Type consistency:** `Settings` fields, `Base`/`engine`/`SessionLocal`/`get_db` (db.py), model class names, `save_bytes`/`path_for`/`read_file` (storage.py), and `Patient*` schemas are referenced consistently across tasks.
- **Known constraint:** all DB-touching tests require a reachable Postgres with permission to `CREATE EXTENSION vector`. Supabase grants this on the project DB.
