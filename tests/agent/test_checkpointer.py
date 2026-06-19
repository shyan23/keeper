import pytest
from app.agent import checkpointer as cp


def test_pg_conninfo_strips_sqlalchemy_dialect(monkeypatch):
    monkeypatch.setattr(
        cp.get_settings(), "database_url",
        "postgresql+psycopg://u:p@host:5432/db", raising=True,
    )
    cp.pg_conninfo.cache_clear()
    assert cp.pg_conninfo() == "postgresql://u:p@host:5432/db"


def test_pg_conninfo_leaves_plain_url_unchanged(monkeypatch):
    monkeypatch.setattr(
        cp.get_settings(), "database_url",
        "postgresql://u:p@host:5432/db", raising=True,
    )
    cp.pg_conninfo.cache_clear()
    assert cp.pg_conninfo() == "postgresql://u:p@host:5432/db"
