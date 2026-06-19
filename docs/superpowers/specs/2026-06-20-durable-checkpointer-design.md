# Durable Postgres Checkpointer — Design

**Date:** 2026-06-20
**Status:** Approved
**Horizon:** H2 (Durability, insight, observability) — sub-project 1 of 3
**Sequence:** checkpointer (this) → typed numeric results → self-hosted tracing (Langfuse OSS)

## Problem

The LangGraph supervisor compiles with an in-process `MemorySaver`
(`app/agent/graph.py:135`, `app/api/runtime.py:32`). Thread state — including
mid-flight HITL confirmation gates (ingest approval, edit verify, report
approve/rebuild) — is lost on every process restart. A user who is waiting on a
confirmation prompt when the server restarts cannot resume; the thread is gone.

## Goal

Graph and HITL thread state survives process restart, backed by the existing
Postgres database, and is safe under the concurrent execution model the SSE
layer already uses. Drop-in replacement for `MemorySaver` with no change to node
logic or the graph topology.

## Key constraint: concurrency

`app/api/sse.py:46` runs `graph.stream(...)` inside a **background thread, one
per request**. The graph (and therefore its checkpointer) is a cached singleton
(`runtime.get_graph()` is `@lru_cache`). Concurrent requests therefore drive one
shared checkpointer from multiple threads simultaneously. A single psycopg
`Connection` is **not** safe for concurrent use across threads, so a
`ConnectionPool` is required for correctness, not merely for performance.

## Key constraint: Supabase pooler

`DATABASE_URL` targets `aws-1-...pooler.supabase.com:5432` — the Supavisor
**session** pooler (port 5432, not the 6543 transaction pooler). Session mode
holds a backend per connection and supports prepared statements. We still set
`prepare_threshold=None` so the saver does not rely on server-side prepared
statements, sidestepping the well-known PgBouncer/transaction-pooler failure
mode if the URL is ever pointed at a transaction pooler.

## Design

### 1. Dependency

Add `langgraph-checkpoint-postgres`, pinned to a release compatible with the
installed `langgraph==0.2.60` / `langgraph-checkpoint==2.1.2`. It pulls
`psycopg_pool`. Add to `pyproject.toml`.

### 2. New module `app/agent/checkpointer.py`

```python
def pg_conninfo() -> str:
    """Plain libpq conninfo for the checkpointer DB. Mirrors app/db.py's
    resolution (test_database_url or database_url) so tests target the test DB
    and never production, then strips the SQLAlchemy '+psycopg' dialect tag:
    'postgresql+psycopg://...' -> 'postgresql://...'."""

@lru_cache(maxsize=1)
def get_checkpointer():
    """Build a thread-safe PostgresSaver over a ConnectionPool, run .setup()
    once (idempotent — creates the checkpoints* tables), return it."""
    pool = ConnectionPool(
        conninfo=pg_conninfo(),
        kwargs={"autocommit": True, "row_factory": dict_row,
                "prepare_threshold": None},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return saver
```

`get_checkpointer()` is cached, so the pool and `setup()` run once per process.
`prepare_threshold=None`, `autocommit=True`, and `row_factory=dict_row` are the
configuration langgraph's PostgresSaver requires.

### 3. Wire-in `app/api/runtime.py`

`get_graph()` calls `build_graph(checkpointer=get_checkpointer())`.

`build_graph` keeps its existing `checkpointer or MemorySaver()` fallback
(`graph.py:135`) unchanged, so all existing tests that call `build_graph()` with
no argument continue to run fully in-memory and are untouched.

**Degrade, do not crash:** `get_graph()` wraps `get_checkpointer()` in a
try/except. On any connect/`setup()` error it logs a warning and builds the
graph with `MemorySaver` (non-durable, but the app still serves requests). This
was confirmed over hard-fail: a transient DB hiccup at boot should not take the
whole chat service down.

### 4. Migrations

None in Alembic. `PostgresSaver.setup()` creates and version-manages its own
tables (`checkpoints`, `checkpoint_writes`, `checkpoint_blobs`,
`checkpoint_migrations`). Documented here so these out-of-Alembic tables are not
a surprise during DB inspection. They live in the same database as the
application records, which is acceptable: thread state already contains the same
class of medical content as the records themselves.

## Testing

- **Unit** — `pg_conninfo()` strips `+psycopg` and yields a libpq-form URL;
  also verify it leaves an already-plain `postgresql://` URL unchanged.
- **Integration** (local pgvector docker on `:5433`, `TEST_DATABASE_URL`) —
  build a graph with a `PostgresSaver`, advance a thread to a state, then build
  a *fresh* graph instance with the same `thread_id` and assert `get_state`
  returns the previously persisted state. This proves durability across a
  simulated process restart (new graph object, same backing store).
- Existing 200+ tests: unaffected (they use the `MemorySaver` default path).

## Out of scope (deferred to later H2 sub-projects)

- Typed numeric test results + trend backing (sub-project 2).
- Self-hosted tracing via Langfuse OSS (sub-project 3).
- At-rest encryption of `STORAGE_DIR` (separate H1/H2 item in the roadmap).
- Connection-pool tuning / sizing knobs beyond psycopg defaults — add only if
  contention is observed under real load.
