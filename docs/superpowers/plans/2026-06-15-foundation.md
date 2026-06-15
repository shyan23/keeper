# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a runnable Python service/data layer backed by Supabase(Postgres/pgvector) with the full DB schema, local-disk file storage, a testable patient service, and a health check — the scaffold every later sub-project builds on. No UI yet.

**Architecture:** Plain Python modules (config, db, models, storage, services), fully unit-tested. SQLAlchemy 2.x models + Alembic migrations against Supabase Postgres (reached over the IPv4 session pooler) with the `vector` extension. Raw files live on local disk under a patient-scoped layout; the DB stores only the path + metadata.

**Tech Stack:** Python 3.11+, SQLAlchemy 2.x, Alembic, psycopg (v3), pgvector, pydantic-settings, pytest.

---

## File Structure

```
app/
  __init__.py
  config.py             # pydantic-settings: DATABASE_URL, GEMINI_API_KEY, STORAGE_DIR, APP_VERSION
  db.py                 # SQLAlchemy engine + session factory + Base
  models.py             # all ORM models (patient, document, entities, links, chunk)
  storage.py            # local-disk file helpers: save_bytes / path_for / read_file
  services/
    __init__.py
    patients.py         # patient CRUD functions
    health.py           # check_health()
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
alembic.ini
.env.example
requirements.txt
README.md
```

**Test DB:** DB-touching tests run against `TEST_DATABASE_URL` (the Supabase session-pooler URL in `.env`). `conftest.py` ensures the `vector` extension + tables exist and cleans rows per test. No mocking of Postgres.

---

### Task 1: Scaffold + dependencies + config

**Files:**
- Create: `requirements.txt`, `.env.example`
- Create: `app/__init__.py`, `app/config.py`, `tests/__init__.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write `requirements.txt`**

```
sqlalchemy==2.0.36
alembic==1.14.0
psycopg[binary]==3.2.3
pgvector==0.3.6
pydantic-settings==2.7.0
pytest==8.3.4
```

- [ ] **Step 2: Write `.env.example`**

```
# Supabase session pooler (IPv4). user is postgres.<project_ref>
DATABASE_URL=postgresql+psycopg://postgres.[PROJECT]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
TEST_DATABASE_URL=postgresql+psycopg://postgres.[PROJECT]:[PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres
GEMINI_API_KEY=changeme
STORAGE_DIR=./data/files
APP_VERSION=0.1.0
```

- [ ] **Step 3: Create empty `app/__init__.py` and `tests/__init__.py`**

- [ ] **Step 4: Write the failing test** — `tests/test_config.py`

```python
from app.config import Settings


def test_settings_reads_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
    monkeypatch.setenv("GEMINI_API_KEY", "key123")
    monkeypatch.setenv("STORAGE_DIR", "/tmp/files")
    s = Settings()
    assert s.database_url == "postgresql+psycopg://u:p@h:5432/db"
    assert s.gemini_api_key == "key123"
    assert s.storage_dir == "/tmp/files"


def test_settings_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@h:5432/db")
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

Run: `pytest tests/test_db.py -v`
Expected: PASS (2 passed). Uses `TEST_DATABASE_URL` from `.env` (Supabase pooler).

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


def test_document_has_file_path():
    cols = {c.name for c in Document.__table__.columns}
    assert "file_path" in cols
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
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)  # local disk path under STORAGE_DIR
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
        sa.Column("file_path", sa.Text),
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
        assert "file_path" in cols
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_migration.py -v`
Expected: PASS once the migration files exist (before then, alembic errors → FAIL).

- [ ] **Step 6: Commit**

```bash
git add alembic.ini migrations/ tests/test_migration.py
git commit -m "feat: alembic initial migration with full schema and pgvector"
```

---

### Task 5: Local-disk file storage helpers

**Files:**
- Create: `app/storage.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write the failing test** — `tests/test_storage.py`

```python
from pathlib import Path

import app.storage as storage
from app.storage import save_bytes, path_for, read_file


def test_path_for(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    assert path_for(1, 2, "png") == str(tmp_path / "1" / "2.png")
    assert path_for(1, 2, ".jpg") == str(tmp_path / "1" / "2.jpg")


def test_save_and_read(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    p = save_bytes(patient_id=7, document_id=42, ext="pdf", data=b"%PDF-1.4 hi")
    assert Path(p).exists()
    assert p == str(tmp_path / "7" / "42.pdf")
    assert read_file(p) == b"%PDF-1.4 hi"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.storage'`

- [ ] **Step 3: Implement `app/storage.py`**

```python
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
git commit -m "feat: local-disk file storage helpers with patient-scoped layout"
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

Run: `pytest tests/test_patients_service.py -v`
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

Run: `pytest tests/test_patients_service.py -v`
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

Run: `pytest tests/test_health_service.py -v`
Expected: PASS (1 passed). Requires the `vector` extension present (conftest creates it).

- [ ] **Step 10: Commit**

```bash
git add app/services/ tests/conftest.py tests/test_patients_service.py tests/test_health_service.py
git commit -m "feat: patient service + health service with tests"
```

---

### Task 7: README + full suite green

**Files:**
- Create: `README.md`
- Test: full `pytest` run

- [ ] **Step 1: Create `README.md`**

```markdown
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

See `docs/superpowers/specs/2026-06-15-medical-doc-intelligence-design.md`.
This repo currently implements sub-project 1 (Foundation).
```

- [ ] **Step 2: Run the full suite**

Run: `pytest -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with setup and test instructions"
```

---

## Self-Review Notes

- **Spec coverage:** scaffold + config (T1), DB engine via pooler (T2), full schema models with `file_path` (T3), Alembic migration + pgvector + `vector(768)` (T4), local-disk storage helpers (T5), patient service + health service (T6), README/success criteria (T7). All Foundation deliverables and success criteria mapped.
- **No UI / no Streamlit / no FastAPI:** intentionally deferred per decision. Foundation is a pure, testable service layer.
- **Type consistency:** `Settings` fields; `engine`/`SessionLocal`/`Base` (db.py); model class names incl. `Document.file_path`; storage funcs `path_for`/`save_bytes`/`read_file` (module-level `STORAGE_DIR` patched in tests); service funcs `create_patient`/`list_patients`/`get_patient`/`update_patient`/`delete_patient`; `check_health` — all referenced consistently across tasks.
- **Out of scope (later sub-projects):** OCR, Gemini calls, entity extraction, chunking/embeddings population, UI (the 5 screens), HITL editing.
- **Known constraint:** DB-touching tests need the Supabase pooler reachable with permission to `CREATE EXTENSION vector` (Supabase grants this on the project DB).
```
