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
def _disable_cache(monkeypatch):
    # Force cache misses so tests never read a real Redis (deterministic, and
    # they don't depend on / pollute a running server).
    monkeypatch.setattr("app.cache._client", lambda: None)
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


@pytest.fixture
def db_session_factory():
    from app.db import SessionLocal
    return SessionLocal
