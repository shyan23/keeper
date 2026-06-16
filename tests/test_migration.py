import subprocess
import sys

from sqlalchemy import text
from app.db import engine

# App tables + alembic's own bookkeeping table. Dropped before the test so the
# migration runs against a clean slate regardless of conftest's create_all().
_TABLES = [
    "chunk", "document_entity", "test_result", "medical_test", "medication",
    "symptom", "disease", "doctor", "document", "patient", "alembic_version",
]


def _run(args):
    # Invoke alembic via the current interpreter so it is found regardless of PATH
    # (a bare `alembic` shell command is not on PATH under every runner).
    return subprocess.run([sys.executable, "-m", *args], check=True,
                          capture_output=True, text=True)


def test_migration_builds_schema_and_vector():
    # conftest's session fixture create_all()s the schema outside alembic's
    # knowledge; drop it so `alembic upgrade head` builds from scratch.
    with engine.begin() as conn:
        for t in _TABLES:
            conn.execute(text(f'DROP TABLE IF EXISTS "{t}" CASCADE'))

    _run(["alembic", "upgrade", "head"])

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
