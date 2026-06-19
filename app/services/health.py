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
