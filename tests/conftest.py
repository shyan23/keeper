import pytest

from app.config import get_settings

# DATA-SAFETY GUARD. The autouse fixtures below DELETE all rows (cascading to
# documents + entities). If the test DB is the production DB, every `pytest`
# run wipes real patient data — which is exactly how live data was lost once.
# Refuse to run unless TEST_DATABASE_URL is set AND distinct from DATABASE_URL.
_settings = get_settings()
_test_db = (_settings.test_database_url or "").strip()
_prod_db = (_settings.database_url or "").strip()
if not _test_db or _test_db == _prod_db:
    raise RuntimeError(
        "Refusing to run the test suite: TEST_DATABASE_URL must be set to a "
        "database DISTINCT from DATABASE_URL. The autouse fixtures delete all "
        "rows, so pointing tests at the production database wipes real patient "
        "data. Set TEST_DATABASE_URL to a throwaway/test database, then re-run."
    )

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
    """Isolate each test WITHOUT wiping pre-existing data. Snapshot the patient
    ids present before the test, then after it remove ONLY the patients the test
    created (cascading to their documents/entities). Original/sample data that
    predates the test is never touched."""
    from app.models import Patient
    db = SessionLocal()
    try:
        preexisting = {pid for (pid,) in db.query(Patient.id).all()}
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        q = db.query(Patient)
        if preexisting:
            q = q.filter(Patient.id.notin_(preexisting))
        # one-by-one so ORM cascade (delete-orphan) fires for docs/entities
        for p in q.all():
            db.delete(p)
        db.commit()
    finally:
        db.close()


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
