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
