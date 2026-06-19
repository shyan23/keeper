import pytest
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from app.agent import checkpointer as cp
from app.config import get_settings

# Read via pydantic settings (loads .env) — NOT os.getenv, which is unset here.
_TEST_DB = get_settings().test_database_url
pg = pytest.mark.skipif(not _TEST_DB, reason="test_database_url not set")


def _saver(conninfo: str) -> PostgresSaver:
    pool = ConnectionPool(
        conninfo=conninfo,
        kwargs={"autocommit": True, "prepare_threshold": None},
        open=True,
    )
    s = PostgresSaver(pool)
    s.setup()
    return s


def test_pg_conninfo_strips_sqlalchemy_dialect(monkeypatch):
    # test_database_url=None so the database_url branch is exercised.
    monkeypatch.setattr(cp.get_settings(), "test_database_url", None, raising=True)
    monkeypatch.setattr(
        cp.get_settings(), "database_url",
        "postgresql+psycopg://u:p@host:5432/db", raising=True,
    )
    cp.pg_conninfo.cache_clear()
    assert cp.pg_conninfo() == "postgresql://u:p@host:5432/db"


def test_pg_conninfo_leaves_plain_url_unchanged(monkeypatch):
    monkeypatch.setattr(cp.get_settings(), "test_database_url", None, raising=True)
    monkeypatch.setattr(
        cp.get_settings(), "database_url",
        "postgresql://u:p@host:5432/db", raising=True,
    )
    cp.pg_conninfo.cache_clear()
    assert cp.pg_conninfo() == "postgresql://u:p@host:5432/db"


def test_pg_conninfo_prefers_test_database_url(monkeypatch):
    # In test/CI the checkpointer must target the test DB, never production.
    monkeypatch.setattr(
        cp.get_settings(), "test_database_url",
        "postgresql+psycopg://t:t@testhost:5433/test", raising=True,
    )
    monkeypatch.setattr(
        cp.get_settings(), "database_url",
        "postgresql+psycopg://prod:prod@prodhost:5432/prod", raising=True,
    )
    cp.pg_conninfo.cache_clear()
    assert cp.pg_conninfo() == "postgresql://t:t@testhost:5433/test"


@pg
def test_state_survives_restart():
    conninfo = _TEST_DB.replace("postgresql+psycopg://", "postgresql://", 1)
    thread = "durability-test-thread"
    config = {"configurable": {"thread_id": thread, "checkpoint_ns": ""}}

    # First "process": write a checkpoint.
    saver_a = _saver(conninfo)
    saver_a.put(
        config,
        {"v": 1, "ts": "t0", "id": "c0", "channel_values": {"x": 42},
         "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )

    # Second "process": a fresh saver over the same store sees it.
    saver_b = _saver(conninfo)
    got = saver_b.get_tuple(config)
    assert got is not None
    assert got.checkpoint["channel_values"]["x"] == 42


@pg
def test_get_checkpointer_builds_and_persists(monkeypatch):
    monkeypatch.setattr(cp.get_settings(), "database_url", _TEST_DB, raising=True)
    cp.pg_conninfo.cache_clear()
    cp.get_checkpointer.cache_clear()
    saver = cp.get_checkpointer()
    config = {"configurable": {"thread_id": "factory-test-thread", "checkpoint_ns": ""}}
    saver.put(
        config,
        {"v": 1, "ts": "t0", "id": "c0", "channel_values": {"y": 7},
         "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )
    assert cp.get_checkpointer().get_tuple(config).checkpoint["channel_values"]["y"] == 7
    cp.get_checkpointer.cache_clear()


from langgraph.checkpoint.memory import MemorySaver
from app.api import runtime


def test_get_graph_degrades_to_memorysaver_on_error(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(runtime, "get_checkpointer", boom)
    runtime.get_graph.cache_clear()
    graph = runtime.get_graph()
    assert isinstance(graph.checkpointer, MemorySaver)
    runtime.get_graph.cache_clear()
