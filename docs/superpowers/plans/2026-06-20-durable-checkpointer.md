# Durable Postgres Checkpointer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-process `MemorySaver` with a Postgres-backed LangGraph checkpointer so graph/HITL thread state survives process restarts and is safe under concurrent SSE threads.

**Architecture:** A cached `ConnectionPool` (thread-safe) wrapped in `PostgresSaver` lives in a new `app/agent/checkpointer.py`. `runtime.get_graph()` injects it, degrading to `MemorySaver` if the DB is unreachable at boot. `build_graph`'s existing `or MemorySaver()` fallback keeps all current tests in-memory.

**Tech Stack:** LangGraph 0.2.60, `langgraph-checkpoint-postgres==2.0.25`, psycopg3 + psycopg_pool, Postgres (Supabase session pooler in prod, local pgvector docker `:5433` for tests).

**Spec:** `docs/superpowers/specs/2026-06-20-durable-checkpointer-design.md`

---

## File Structure

- `requirements.txt` — add the postgres checkpointer dependency.
- `app/agent/checkpointer.py` (new) — `pg_conninfo()` URL helper + cached `get_checkpointer()` pool/saver factory. Single responsibility: build the durable saver.
- `app/api/runtime.py` (modify) — `get_graph()` injects the saver with degrade-to-MemorySaver fallback.
- `tests/agent/test_checkpointer.py` (new) — unit (conninfo) + integration (durability across restart) + fallback tests.

---

### Task 1: Add the dependency

**Files:**
- Modify: `requirements.txt:10`

- [ ] **Step 1: Add the pinned line**

In `requirements.txt`, directly under `langgraph==0.2.60` (line 10), add:

```
langgraph-checkpoint-postgres==2.0.25
```

- [ ] **Step 2: Install and verify no langgraph upgrade**

Run: `.venv/bin/pip install langgraph-checkpoint-postgres==2.0.25`
Expected: installs `langgraph-checkpoint-postgres-2.0.25` and `psycopg-pool-3.3.1`; `langgraph` stays at `0.2.60`.

Verify: `.venv/bin/python -c "from langgraph.checkpoint.postgres import PostgresSaver; from psycopg_pool import ConnectionPool; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "build(h2): add langgraph-checkpoint-postgres for durable saver"
```

---

### Task 2: `pg_conninfo()` URL helper (TDD)

Derives a plain libpq conninfo string from the SQLAlchemy `database_url` by
stripping the `+psycopg` dialect tag. psycopg/langgraph reject the
`postgresql+psycopg://` form.

**Files:**
- Create: `app/agent/checkpointer.py`
- Test: `tests/agent/test_checkpointer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/agent/test_checkpointer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py -v`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module 'app.agent.checkpointer' has no attribute 'pg_conninfo'`.

- [ ] **Step 3: Write minimal implementation**

Create `app/agent/checkpointer.py`:

```python
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def pg_conninfo() -> str:
    """Plain libpq conninfo from settings.database_url.

    Strips the SQLAlchemy '+psycopg' dialect tag so psycopg/langgraph accept
    it: 'postgresql+psycopg://...' -> 'postgresql://...'.
    """
    # Mirror app/db.py: test DB in tests, prod otherwise. Never prod in tests.
    s = get_settings()
    url = s.test_database_url or s.database_url
    return url.replace("postgresql+psycopg://", "postgresql://", 1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add app/agent/checkpointer.py tests/agent/test_checkpointer.py
git commit -m "feat(h2): pg_conninfo helper for durable checkpointer"
```

---

### Task 3: `get_checkpointer()` factory + durability integration test (TDD)

Builds the thread-safe pool + `PostgresSaver`, runs `.setup()` once. The test
proves state survives a simulated restart (a fresh saver over the same store,
same `thread_id`).

**Files:**
- Modify: `app/agent/checkpointer.py`
- Test: `tests/agent/test_checkpointer.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/agent/test_checkpointer.py`:

```python
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
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


@pg
def test_state_survives_restart():
    conninfo = _TEST_DB.replace("postgresql+psycopg://", "postgresql://", 1)
    thread = "durability-test-thread"
    config = {"configurable": {"thread_id": thread}}

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py::test_state_survives_restart -v`
Expected: PASS if `TEST_DATABASE_URL` points at a running Postgres; SKIP if unset. (This test exercises the library directly to lock the persistence contract before `get_checkpointer` exists.) If it errors connecting, start the test DB: `docker ps` should show the pgvector container on `:5433`.

- [ ] **Step 3: Write the `get_checkpointer` implementation**

Append to `app/agent/checkpointer.py`:

```python
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver


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
```

- [ ] **Step 4: Add a test that the factory builds and persists**

Append to `tests/agent/test_checkpointer.py`:

```python
@pg
def test_get_checkpointer_builds_and_persists(monkeypatch):
    monkeypatch.setattr(cp.get_settings(), "database_url", _TEST_DB, raising=True)
    cp.pg_conninfo.cache_clear()
    cp.get_checkpointer.cache_clear()
    saver = cp.get_checkpointer()
    config = {"configurable": {"thread_id": "factory-test-thread"}}
    saver.put(
        config,
        {"v": 1, "ts": "t0", "id": "c0", "channel_values": {"y": 7},
         "channel_versions": {}, "versions_seen": {}, "pending_sends": []},
        {"source": "input", "step": 0, "writes": {}, "parents": {}},
        {},
    )
    assert cp.get_checkpointer().get_tuple(config).checkpoint["channel_values"]["y"] == 7
    cp.get_checkpointer.cache_clear()
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py -v`
Expected: PASS (or SKIP if `TEST_DATABASE_URL` unset).

- [ ] **Step 6: Commit**

```bash
git add app/agent/checkpointer.py tests/agent/test_checkpointer.py
git commit -m "feat(h2): durable PostgresSaver factory (get_checkpointer)"
```

---

### Task 4: Wire `runtime.get_graph()` with degrade-to-MemorySaver

`get_graph()` injects the durable saver. On any connect/setup error it logs a
warning and falls back to `MemorySaver` so a transient DB hiccup at boot does
not take down the chat service.

**Files:**
- Modify: `app/api/runtime.py:31-35`
- Test: `tests/agent/test_checkpointer.py`

- [ ] **Step 1: Write the failing fallback test**

Append to `tests/agent/test_checkpointer.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py::test_get_graph_degrades_to_memorysaver_on_error -v`
Expected: FAIL — `AttributeError: module 'app.api.runtime' has no attribute 'get_checkpointer'` (not yet imported there).

- [ ] **Step 3: Modify `runtime.get_graph()`**

In `app/api/runtime.py`, add to the imports near the top:

```python
import logging

from app.agent.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)
```

Replace the existing `get_graph` (lines 31-35):

```python
@lru_cache(maxsize=1)
def get_graph():
    """Compile the LangGraph supervisor once (in-process MemorySaver checkpointer)."""
    from app.agent.graph import build_graph
    return build_graph()
```

with:

```python
@lru_cache(maxsize=1)
def get_graph():
    """Compile the LangGraph supervisor once with a durable Postgres
    checkpointer. Degrades to in-process MemorySaver if the DB is
    unreachable at boot (non-durable, but the service stays up)."""
    from app.agent.graph import build_graph
    try:
        return build_graph(checkpointer=get_checkpointer())
    except Exception:
        logger.warning(
            "Durable checkpointer unavailable; falling back to MemorySaver "
            "(thread state will NOT survive restart).", exc_info=True,
        )
        return build_graph()
```

- [ ] **Step 4: Run the fallback test**

Run: `.venv/bin/pytest tests/agent/test_checkpointer.py::test_get_graph_degrades_to_memorysaver_on_error -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `.venv/bin/pytest -q`
Expected: existing tests pass; new checkpointer tests pass or SKIP (if `TEST_DATABASE_URL` unset). No failures.

- [ ] **Step 6: Commit**

```bash
git add app/api/runtime.py tests/agent/test_checkpointer.py
git commit -m "feat(h2): inject durable checkpointer into get_graph with fallback"
```

---

## Self-Review notes

- **Spec coverage:** dependency (Task 1), `pg_conninfo` (Task 2), `get_checkpointer` + ConnectionPool + setup + durability test (Task 3), runtime wire-in + degrade-to-MemorySaver + unit/integration/fallback tests (Task 4). Migrations: none (covered — `setup()` owns its tables, no Alembic task). All spec sections mapped.
- **Type consistency:** `pg_conninfo()` and `get_checkpointer()` names match across tasks and the runtime import. `MemorySaver` fallback path is `build_graph()` (the existing `or MemorySaver()` default, unchanged).
- **Pooler note:** prod URL is port 5432 (session pooler); `prepare_threshold=None` keeps it safe even against a transaction pooler.
