from __future__ import annotations

import logging
from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def pg_conninfo() -> str:
    """Plain libpq conninfo for the checkpointer's database.

    Mirrors app/db.py's resolution (test_database_url or database_url) so the
    checkpointer lives in the SAME database as the ORM — pointing at the test
    DB during tests, never production. Strips the SQLAlchemy '+psycopg' dialect
    tag so psycopg/langgraph accept it: 'postgresql+psycopg://...' ->
    'postgresql://...'.
    """
    s = get_settings()
    url = s.test_database_url or s.database_url
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache(maxsize=1)
def get_checkpointer() -> PostgresSaver:
    """Thread-safe PostgresSaver over a ConnectionPool, set up once.

    ConnectionPool is required because app/api/sse.py runs graph.stream in a
    background thread per request, all sharing this cached saver. autocommit +
    prepare_threshold=None are required by langgraph's PostgresSaver and avoid
    server-side prepared statements (pooler-safe).
    """
    pool = ConnectionPool(
        conninfo=pg_conninfo(),
        kwargs={"autocommit": True, "row_factory": dict_row,
                "prepare_threshold": None},
        open=True,
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
