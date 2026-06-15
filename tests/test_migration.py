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
