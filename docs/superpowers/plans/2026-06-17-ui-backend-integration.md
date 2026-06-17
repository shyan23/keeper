# MedAgentic UI ↔ Backend Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Streamlit with the `medagentic-dashboard` vanilla-TS UI, wired to the existing Python services + LangGraph agent through a new FastAPI layer that streams live node progress and preserves HITL approval gates over SSE.

**Architecture:** A FastAPI sidecar (`app/api/`) wraps the tested service layer and the LangGraph supervisor. The graph is compiled once at startup with an in-process `MemorySaver` keyed by `thread_id` (mirrors `streamlit_app._graph/_deps/_cfg`). Chat runs the graph in a background thread that pushes node/progress/interrupt/message/error events onto a queue drained by an SSE generator. The frontend keeps its hand-written DOM rendering and swaps mock `data.ts` for a typed `api.ts` client.

**Tech Stack:** FastAPI, uvicorn, sse (hand-rolled `text/event-stream`), python-multipart, SQLAlchemy, LangGraph; Vite + vanilla TypeScript + Tailwind.

**Security note (carry through Tasks 8-9):** chat text, record titles/descriptions (OCR/LLM-derived), document names, and patient names are untrusted. The frontend renders via `innerHTML` (the placeholder's convention), so every interpolated *dynamic* string MUST pass through the `esc()` helper defined in Task 9. Static class strings and trusted constants do not.

---

## File Structure

**Backend (new package `app/api/`):**
- `app/api/__init__.py` — package marker
- `app/api/schemas.py` — Pydantic response/request models
- `app/api/mapping.py` — pure DB-row → UI-shape functions
- `app/api/runtime.py` — graph/deps singletons, `cfg()` builder, `NODE_LABELS`
- `app/api/sse.py` — SSE formatting + the queue/thread graph driver
- `app/api/routes_browse.py` — health, patients, records, documents
- `app/api/routes_chat.py` — `/chat/upload`, `/chat/stream`, `/chat/resume`
- `app/api/server.py` — FastAPI app, CORS, router mounts

**Frontend (`medagentic-dashboard/src/`):**
- `src/api.ts` — typed client + SSE reader (new)
- `src/types.ts` — extend with API shapes
- `src/main.ts` — rewired to call the API (replaces mock usage)
- `src/data.ts` — deleted

**Tests:** `tests/test_api_mapping.py`, `tests/test_api_browse.py`, `tests/test_api_chat.py`

**Removed:** `streamlit_app.py`; `streamlit` dep.

---

## Task 1: Dependencies + remove Streamlit

**Files:**
- Modify: `requirements.txt`
- Delete: `streamlit_app.py`

- [ ] **Step 1: Edit `requirements.txt`** — remove the streamlit line, add API deps.

Replace the line `streamlit==1.41.1` with:

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
python-multipart==0.0.20
```

- [ ] **Step 2: Install**

Run: `pip install -r requirements.txt`
Expected: installs fastapi, uvicorn, python-multipart; no errors.

- [ ] **Step 3: Delete the Streamlit app**

Run: `git rm streamlit_app.py`
Expected: `rm 'streamlit_app.py'`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore: drop Streamlit, add FastAPI deps"
```

---

## Task 2: Pydantic schemas

**Files:**
- Create: `app/api/__init__.py`
- Create: `app/api/schemas.py`
- Test: `tests/test_api_mapping.py` (shared file; this task adds the import smoke test)

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_mapping.py`:

```python
from app.api.schemas import (
    DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut,
)


def test_patient_out_serializes():
    p = PatientOut(id="1", name="Jane", age=42, gender="female",
                   bloodType="—", image="http://x", lastVisit="2026-06-01",
                   status="Active")
    assert p.model_dump()["id"] == "1"


def test_patient_in_optional_fields():
    p = PatientIn(name="Jane")
    assert p.age is None and p.gender is None and p.relationship is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_mapping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api'`

- [ ] **Step 3: Create the package + schemas**

Create empty `app/api/__init__.py`.

Create `app/api/schemas.py`:

```python
from __future__ import annotations

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    db: str
    pgvector: bool
    version: str


class PatientOut(BaseModel):
    id: str
    name: str
    age: int | None = None
    gender: str | None = None
    bloodType: str = "—"
    image: str
    lastVisit: str
    status: str = "Active"


class PatientIn(BaseModel):
    name: str
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None


class RecordOut(BaseModel):
    id: str
    patientId: str
    type: str  # disease | symptom | medicine | test_result | treatment_plan
    title: str
    description: str
    date: str | None = None
    status: str = "Recorded"
    severity: str | None = None
    doctor: str | None = None


class DocumentOut(BaseModel):
    id: str
    name: str
    date: str | None = None
    type: str
    size: str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_mapping.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add app/api/__init__.py app/api/schemas.py tests/test_api_mapping.py
git commit -m "feat(api): pydantic schemas for UI shapes"
```

---

## Task 3: Pure mapping functions

**Files:**
- Create: `app/api/mapping.py`
- Test: `tests/test_api_mapping.py` (append)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api_mapping.py`:

```python
from app.api import mapping


def test_avatar_url_encodes_name():
    url = mapping.avatar_url("Jane Doe")
    assert url == "https://i.pravatar.cc/150?u=Jane%20Doe"


def test_patient_to_out_derives_fields():
    class P:  # minimal stand-in for the Patient ORM row
        id = 7
        name = "Jane Doe"
        age = 30
        gender = "female"
    out = mapping.patient_to_out(P(), last_visit="2026-06-10")
    assert out["id"] == "7"
    assert out["bloodType"] == "—"
    assert out["image"] == "https://i.pravatar.cc/150?u=Jane%20Doe"
    assert out["lastVisit"] == "2026-06-10"
    assert out["status"] == "Active"


def test_merge_records_maps_types_and_titles():
    diseases = [{"name": "Diabetes", "source": "dx span", "doc_type": "lab",
                 "date": "2026-05-01", "document_id": 3}]
    symptoms = [{"name": "Fatigue", "source": None, "doc_type": None,
                 "date": "2026-05-01", "document_id": 3}]
    meds = [{"name": "Metformin", "source": "rx", "doc_type": "rx",
             "date": "2026-05-02", "document_id": 4}]
    tests = [{"test": "HbA1c", "value": "6.8", "unit": "%", "source": "lab span",
              "doc_type": "lab", "date": "2026-05-01", "document_id": 3}]
    rows = mapping.merge_records("1", diseases, symptoms, meds, tests)
    by_type = {r["type"] for r in rows}
    assert by_type == {"disease", "symptom", "medicine", "test_result"}
    med = next(r for r in rows if r["type"] == "medicine")
    assert med["title"] == "Metformin"
    tr = next(r for r in rows if r["type"] == "test_result")
    assert tr["title"] == "HbA1c: 6.8%"
    assert all(r["patientId"] == "1" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)  # ids unique


def test_merge_records_empty():
    assert mapping.merge_records("1", [], [], [], []) == []


def test_document_to_out_basename_and_size():
    row = {"id": 9, "file": "/data/files/3/9.pdf", "type": "lab",
           "date": "2026-06-01 10:00"}
    out = mapping.document_to_out(row, size_str="1.2 MB")
    assert out["id"] == "9"
    assert out["name"] == "9.pdf"
    assert out["type"] == "lab"
    assert out["size"] == "1.2 MB"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_mapping.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.mapping'`

- [ ] **Step 3: Create `app/api/mapping.py`**

```python
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

# browse entity_type -> UI record type
_ENTITY_TO_UI = {"disease": "disease", "symptom": "symptom", "medication": "medicine"}


def avatar_url(name: str) -> str:
    return f"https://i.pravatar.cc/150?u={quote(name or '')}"


def patient_to_out(patient, last_visit: str | None) -> dict:
    return {
        "id": str(patient.id),
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "bloodType": "—",
        "image": avatar_url(patient.name),
        "lastVisit": last_visit or "",
        "status": "Active",
    }


def _record(patient_id: str, ui_type: str, idx: int, title: str,
            row: dict) -> dict:
    return {
        "id": f"{ui_type}-{row.get('document_id')}-{idx}",
        "patientId": patient_id,
        "type": ui_type,
        "title": title,
        "description": (row.get("source") or row.get("doc_type") or ""),
        "date": row.get("date"),
        "status": "Recorded",
        "severity": None,
        "doctor": None,
    }


def merge_records(patient_id: str, diseases: list[dict], symptoms: list[dict],
                  medications: list[dict], tests: list[dict]) -> list[dict]:
    out: list[dict] = []
    idx = 0
    for ui_type, rows in (("disease", diseases), ("symptom", symptoms),
                          ("medicine", medications)):
        for r in rows:
            out.append(_record(patient_id, ui_type, idx, r.get("name") or "", r))
            idx += 1
    for r in tests:
        value = " ".join(x for x in [r.get("value"), r.get("unit")] if x)
        title = f"{r.get('test') or ''}: {value}".strip().rstrip(":")
        out.append(_record(patient_id, "test_result", idx, title, r))
        idx += 1
    return out


def document_to_out(row: dict, size_str: str) -> dict:
    file_path = row.get("file") or ""
    name = Path(file_path).name if file_path else f"document-{row.get('id')}"
    return {
        "id": str(row.get("id")),
        "name": name,
        "date": row.get("date"),
        "type": (row.get("type") or "FILE"),
        "size": size_str,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_mapping.py -v`
Expected: PASS (all mapping tests green)

- [ ] **Step 5: Commit**

```bash
git add app/api/mapping.py tests/test_api_mapping.py
git commit -m "feat(api): pure DB-row to UI-shape mappers"
```

---

## Task 4: Runtime singletons + node labels

**Files:**
- Create: `app/api/runtime.py`
- Test: `tests/test_api_chat.py` (created here, asserts labels + cfg shape only — no graph build)

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_chat.py`:

```python
from app.api import runtime


def test_node_labels_cover_key_nodes():
    for node in ["router", "extract_text", "generate_answer"]:
        assert node in runtime.NODE_LABELS


def test_cfg_has_thread_and_progress():
    calls = []
    cfg = runtime.cfg("thread-abc", deps={"x": 1}, progress=calls.append)
    assert cfg["configurable"]["thread_id"] == "thread-abc"
    assert cfg["configurable"]["deps"] == {"x": 1}
    cfg["configurable"]["progress"]("hi")
    assert calls == ["hi"]
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_chat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.runtime'`

- [ ] **Step 3: Create `app/api/runtime.py`**

```python
from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

# Maps graph node keys to user-facing progress labels (ported from streamlit_app).
NODE_LABELS: dict[str, str] = {
    "router": "🧭 Routing your request…",
    "extract_text": "📖 Reading the document (OCR)…",
    "extract_entities": "🔬 Extracting patient, symptoms, meds, tests…",
    "resolve_patient": "🧑 Matching patient…",
    "persist": "💾 Saving entities…",
    "chunk_embed": "📚 Indexing for search…",
    "parse_filters": "🔎 Parsing your query…",
    "query_db": "🗂️ Looking up records…",
    "transform_query": "✍️ Reformulating the query…",
    "retrieve": "🔍 Searching documents…",
    "rerank": "📊 Ranking results…",
    "grade": "⚖️ Checking answer confidence…",
    "correct_query": "🔁 Refining the search…",
    "generate_answer": "🧠 Composing the answer…",
}


@lru_cache(maxsize=1)
def get_graph():
    """Compile the LangGraph supervisor once (in-process MemorySaver checkpointer)."""
    from app.agent.graph import build_graph
    return build_graph()


@lru_cache(maxsize=1)
def get_deps():
    """Build agent Deps once (probes the embedder over the network)."""
    from app.agent.providers import build_deps
    from app.db import SessionLocal
    return build_deps(SessionLocal)


def cfg(thread_id: str, deps: Any = None,
        progress: Callable[[str], None] | None = None) -> dict:
    configurable: dict[str, Any] = {
        "deps": deps if deps is not None else get_deps(),
        "thread_id": thread_id,
    }
    if progress is not None:
        configurable["progress"] = progress
    return {"configurable": configurable}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_chat.py -v`
Expected: PASS (2 passed). Note: `get_deps`/`get_graph` are NOT called here (no network).

- [ ] **Step 5: Commit**

```bash
git add app/api/runtime.py tests/test_api_chat.py
git commit -m "feat(api): graph/deps singletons + node labels"
```

---

## Task 5: SSE driver (queue + background thread)

**Files:**
- Create: `app/api/sse.py`
- Test: `tests/test_api_chat.py` (append)

The driver runs `graph.stream` (a blocking iterator) in a background thread so the
progress callback — which fires *inside* a node — can push events while the request
handler streams them. Events land on a `queue.Queue`; the SSE generator drains it.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api_chat.py`:

```python
from langgraph.types import Command

from app.api import sse


class _FakeInterrupt:
    def __init__(self, value):
        self.value = value


class FakeGraph:
    """Scriptable stand-in for the compiled LangGraph."""

    def __init__(self, chunks, final_messages=None):
        self._chunks = chunks
        self._final = final_messages or []
        self.last_input = None

    def stream(self, payload, cfg, stream_mode="updates"):
        self.last_input = payload
        for ch in self._chunks:
            yield ch

    def get_state(self, cfg):
        class S:
            values = {"messages": self._final}
        return S()


def _collect(gen):
    return "".join(line for line in gen if line.strip())


def test_sse_clean_answer_sequence():
    graph = FakeGraph(
        chunks=[{"router": {}}, {"generate_answer": {}}],
        final_messages=[{"role": "assistant", "content": "Hi", "sources": ["a.pdf"]}],
    )
    body = _collect(sse.run_graph_sse(graph, {"messages": []}, "t1", deps={}))
    assert "event: node" in body
    assert "🧠 Composing the answer" in body
    assert "event: message" in body
    assert "Hi" in body
    assert "a.pdf" in body
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_sse_interrupt_then_resume():
    graph = FakeGraph(
        chunks=[{"extract_text": {}},
                {"__interrupt__": (_FakeInterrupt({"type": "confirm_ingest",
                                                   "extracted": {}}),)}],
    )
    body = _collect(sse.run_graph_sse(graph, {"messages": []}, "t2", deps={}))
    assert "event: interrupt" in body
    assert "confirm_ingest" in body
    assert "event: message" not in body  # paused: no final message

    graph2 = FakeGraph(chunks=[{"persist": {}}],
                       final_messages=[{"role": "assistant", "content": "done"}])
    body2 = _collect(sse.run_graph_sse(graph2, Command(resume={"approved": True}),
                                       "t2", deps={}))
    assert "💾 Saving entities" in body2
    assert "event: message" in body2
    assert isinstance(graph2.last_input, Command)


def test_sse_error_event():
    class Boom(FakeGraph):
        def stream(self, payload, cfg, stream_mode="updates"):
            raise RuntimeError("provider down")
            yield  # pragma: no cover

    body = _collect(sse.run_graph_sse(Boom([]), {"messages": []}, "t3", deps={}))
    assert "event: error" in body
    assert "provider down" in body
    assert "event: done" in body
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_chat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.sse'`

- [ ] **Step 3: Create `app/api/sse.py`**

```python
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Iterator

from app.api import runtime

_DONE = object()  # sentinel: graph thread finished, stop draining


def sse_event(event: str, data: dict | None = None) -> str:
    return f"event: {event}\ndata: {json.dumps(data or {}, default=str)}\n\n"


def run_graph_sse(graph, payload: Any, thread_id: str,
                  deps: Any = None) -> Iterator[str]:
    """Drive graph.stream in a background thread; yield SSE lines.

    `payload` is either an initial state dict or a langgraph Command (resume).
    """
    q: queue.Queue = queue.Queue()

    def progress(msg: str) -> None:
        q.put(("progress", {"msg": msg}))

    def worker() -> None:
        cfg = runtime.cfg(thread_id, deps=deps, progress=progress)
        interrupted = False
        try:
            for chunk in graph.stream(payload, cfg, stream_mode="updates"):
                for node in chunk:
                    if node == "__interrupt__":
                        interrupted = True
                        q.put(("interrupt", chunk["__interrupt__"][0].value))
                        continue
                    q.put(("node", {"label": runtime.NODE_LABELS.get(node, f"… {node}")}))
            if not interrupted:
                snap = graph.get_state(cfg)
                messages = snap.values.get("messages", [])
                last = messages[-1] if messages else None
                if last is not None:
                    q.put(("message", {
                        "role": last.get("role", "assistant"),
                        "content": last.get("content", ""),
                        "sources": last.get("sources"),
                    }))
        except Exception as e:  # noqa: BLE001 - surface to the client as an error event
            q.put(("error", {"message": str(e)}))
        finally:
            q.put((_DONE, None))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, data = q.get()
        if kind is _DONE:
            yield sse_event("done", {})
            return
        yield sse_event(kind, data)
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_api_chat.py -v`
Expected: PASS (all chat-sse tests green)

- [ ] **Step 5: Commit**

```bash
git add app/api/sse.py tests/test_api_chat.py
git commit -m "feat(api): threaded SSE graph driver"
```

---

## Task 6: Browse routes (health, patients, records, documents)

**Files:**
- Create: `app/api/routes_browse.py`
- Create: `app/api/server.py` (minimal app so TestClient can mount routes)
- Test: `tests/test_api_browse.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api_browse.py`:

```python
from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "db" in body and "pgvector" in body and "version" in body


def test_create_and_list_patient():
    r = client.post("/api/patients", json={"name": "Api Tester", "age": 40,
                                           "gender": "female"})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "Api Tester"
    assert created["image"].startswith("https://i.pravatar.cc/")
    assert created["bloodType"] == "—"

    r2 = client.get("/api/patients")
    assert r2.status_code == 200
    names = [p["name"] for p in r2.json()]
    assert "Api Tester" in names


def test_records_and_documents_empty_for_new_patient():
    pid = client.post("/api/patients", json={"name": "Empty One"}).json()["id"]
    assert client.get(f"/api/patients/{pid}/records").json() == []
    assert client.get(f"/api/patients/{pid}/documents").json() == []
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_browse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.server'`

- [ ] **Step 3: Create `app/api/routes_browse.py`**

```python
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.api import mapping
from app.api.schemas import DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut
from app.db import SessionLocal
from app.services import browse as bsvc
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc

router = APIRouter(prefix="/api")


def _last_visit(db: Session, patient_id: int) -> str | None:
    docs = dsvc.list_documents(db, patient_id=patient_id)
    if docs and docs[0].uploaded_at:
        return docs[0].uploaded_at.strftime("%Y-%m-%d")
    p = psvc.get_patient(db, patient_id)
    if p is not None and p.created_at:
        return p.created_at.strftime("%Y-%m-%d")
    return None


@router.get("/health", response_model=HealthOut)
def health() -> dict:
    return hsvc.check_health()


@router.get("/patients", response_model=list[PatientOut])
def list_patients() -> list[dict]:
    db = SessionLocal()
    try:
        return [mapping.patient_to_out(p, _last_visit(db, p.id))
                for p in psvc.list_patients(db)]
    finally:
        db.close()


@router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(body: PatientIn) -> dict:
    db = SessionLocal()
    try:
        p = psvc.create_patient(db, name=body.name, age=body.age,
                                gender=body.gender, relationship=body.relationship)
        return mapping.patient_to_out(p, _last_visit(db, p.id))
    finally:
        db.close()


@router.get("/patients/{patient_id}/records", response_model=list[RecordOut])
def patient_records(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        diseases = bsvc.list_entity_links(db, "disease", patient_id=patient_id)
        symptoms = bsvc.list_entity_links(db, "symptom", patient_id=patient_id)
        meds = bsvc.list_entity_links(db, "medication", patient_id=patient_id)
        tests = bsvc.list_test_results(db, patient_id=patient_id)
        return mapping.merge_records(str(patient_id), diseases, symptoms, meds, tests)
    finally:
        db.close()


@router.get("/patients/{patient_id}/documents", response_model=list[DocumentOut])
def patient_documents(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        rows = bsvc.list_documents_timeline(db, patient_id=patient_id)
        out = []
        for row in rows:
            size_str = "—"
            path = row.get("file")
            try:
                if path and os.path.exists(path):
                    size_str = f"{os.path.getsize(path) / 1_048_576:.1f} MB"
            except OSError:
                pass
            out.append(mapping.document_to_out(row, size_str))
        return out
    finally:
        db.close()
```

- [ ] **Step 4: Create `app/api/server.py`**

```python
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_browse import router as browse_router
from app.api.routes_chat import router as chat_router

app = FastAPI(title="MedAgentic API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev: Vite on :3000; no credentials used
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(browse_router)
app.include_router(chat_router)
```

NOTE: `server.py` imports `routes_chat`, built in Task 7. To run THIS task's test before
Task 7 exists, temporarily comment the two chat lines (`from app.api.routes_chat ...` and
`app.include_router(chat_router)`), run the test, then restore them at the end of Task 7.
Do not commit the commented version.

- [ ] **Step 5: Run to verify pass**

Temporarily comment the chat-router import + include in `server.py`, then:
Run: `pytest tests/test_api_browse.py -v`
Expected: PASS (3 passed). Restore the chat lines after (Task 7 supplies `routes_chat`).

- [ ] **Step 6: Commit**

```bash
git add app/api/routes_browse.py app/api/server.py tests/test_api_browse.py
git commit -m "feat(api): browse routes (health, patients, records, documents)"
```

---

## Task 7: Chat routes (upload, stream, resume)

**Files:**
- Create: `app/api/routes_chat.py`
- Test: `tests/test_api_chat.py` (append integration tests against TestClient)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_api_chat.py`:

```python
import app.api.runtime as runtime_mod
from fastapi.testclient import TestClient


def test_upload_stages_file():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/upload",
                    files={"file": ("scan.png", b"\x89PNG fake", "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ext"] == "png"
    assert body["mime"] == "image/png"
    assert body["staged_path"].endswith(".png")


def test_upload_rejects_unsupported_ext():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/upload",
                    files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400


def test_stream_text_requires_patient():
    from app.api.server import app
    client = TestClient(app)
    r = client.post("/api/chat/stream",
                    json={"thread_id": "t-np", "message": "hello"})
    assert r.status_code == 200
    assert "event: error" in r.text
    assert "pick a patient" in r.text.lower()


def test_stream_runs_graph(monkeypatch):
    from app.api import server
    client = TestClient(server.app)

    class S:
        values = {"messages": [{"role": "assistant", "content": "Answer", "sources": []}]}

    class G:
        def stream(self, payload, cfg, stream_mode="updates"):
            yield {"router": {}}
            yield {"generate_answer": {}}
        def get_state(self, cfg):
            return S()

    monkeypatch.setattr(runtime_mod, "get_graph", lambda: G())
    monkeypatch.setattr(runtime_mod, "get_deps", lambda: {"fake": True})

    r = client.post("/api/chat/stream",
                    json={"thread_id": "t-ok", "message": "hi", "patient_id": 1})
    assert r.status_code == 200
    assert "event: message" in r.text
    assert "Answer" in r.text
    assert r.text.rstrip().endswith("event: done\ndata: {}")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_api_chat.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.api.routes_chat'`

- [ ] **Step 3: Create `app/api/routes_chat.py`**

```python
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel

import app.storage as storage
from app.api import runtime, sse

router = APIRouter(prefix="/api/chat")

_ALLOWED_EXT = {"png", "jpg", "jpeg", "webp", "pdf", "txt"}


def _ext(filename: str) -> str:
    return Path(filename).suffix.lstrip(".").lower() or "bin"


class StreamIn(BaseModel):
    thread_id: str
    message: str | None = None
    patient_id: int | None = None
    staged_path: str | None = None
    mime: str | None = None
    ext: str | None = None


class ResumeIn(BaseModel):
    thread_id: str
    resume: dict


@router.post("/upload")
def upload(file: UploadFile):
    ext = _ext(file.filename or "")
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"unsupported file type: {ext}")
    data = file.file.read()
    staged = storage.save_staging(ext, data)
    mime = file.content_type or (
        "application/pdf" if ext == "pdf"
        else "text/plain" if ext == "txt" else f"image/{ext}")
    return {"staged_path": staged, "mime": mime, "ext": ext}


def _sse_response(gen) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@router.post("/stream")
def stream(body: StreamIn):
    messages = [{"role": "user", "content": body.message or "Read this and arrange it."}]

    if body.staged_path:
        payload = {
            "messages": messages,
            "file_path": body.staged_path,
            "mime_type": body.mime,
            "file_ext": body.ext,
            "source_type": "pdf" if body.ext == "pdf" else "image",
        }
    else:
        if body.patient_id is None:
            def err():
                yield sse.sse_event("error", {
                    "message": "Pick a patient to ask about their records, "
                               "or attach a document for me to read."})
                yield sse.sse_event("done", {})
            return _sse_response(err())
        payload = {"messages": messages, "patient_id": body.patient_id}

    graph = runtime.get_graph()
    return _sse_response(
        sse.run_graph_sse(graph, payload, body.thread_id, deps=runtime.get_deps()))


@router.post("/resume")
def resume(body: ResumeIn):
    graph = runtime.get_graph()
    return _sse_response(
        sse.run_graph_sse(graph, Command(resume=body.resume), body.thread_id,
                          deps=runtime.get_deps()))
```

- [ ] **Step 4: Restore `server.py`** — uncomment the chat-router import + include if you commented them in Task 6. Confirm `app/api/server.py` includes both routers.

- [ ] **Step 5: Run to verify pass**

Run: `pytest tests/test_api_chat.py tests/test_api_browse.py -v`
Expected: PASS (all api tests green). `get_graph`/`get_deps` are monkeypatched in the run-graph test, so no network/LLM is hit.

- [ ] **Step 6: Commit**

```bash
git add app/api/routes_chat.py app/api/server.py tests/test_api_chat.py
git commit -m "feat(api): chat upload/stream/resume over SSE"
```

---

## Task 8: Frontend API client

**Files:**
- Create: `medagentic-dashboard/src/api.ts`
- Modify: `medagentic-dashboard/src/types.ts`
- Modify: `medagentic-dashboard/.env.example`

- [ ] **Step 1: Extend `src/types.ts`** — append API + SSE types:

```typescript
export interface Health {
  status: string;
  db: string;
  pgvector: boolean;
  version: string;
}

export interface ApiPatient {
  id: string;
  name: string;
  age: number | null;
  gender: string | null;
  bloodType: string;
  image: string;
  lastVisit: string;
  status: string;
}

export interface ApiRecord {
  id: string;
  patientId: string;
  type: string;
  title: string;
  description: string;
  date: string | null;
  status: string;
  severity: string | null;
  doctor: string | null;
}

export interface ApiDocument {
  id: string;
  name: string;
  date: string | null;
  type: string;
  size: string;
}

export interface SseHandlers {
  onNode?: (label: string) => void;
  onProgress?: (msg: string) => void;
  onInterrupt?: (payload: any) => void;
  onMessage?: (msg: { role: string; content: string; sources?: string[] }) => void;
  onError?: (message: string) => void;
  onDone?: () => void;
}
```

- [ ] **Step 2: Create `src/api.ts`**

```typescript
import {
  ApiDocument, ApiPatient, ApiRecord, Health, SseHandlers,
} from './types';

const API = (import.meta as any).env?.VITE_API_BASE ?? 'http://localhost:8000';

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => json<Health>('/api/health');
export const listPatients = () => json<ApiPatient[]>('/api/patients');
export const createPatient = (body: {
  name: string; age?: number | null; gender?: string | null; relationship?: string | null;
}) => json<ApiPatient>('/api/patients', { method: 'POST', body: JSON.stringify(body) });
export const getRecords = (patientId: string) =>
  json<ApiRecord[]>(`/api/patients/${patientId}/records`);
export const getDocuments = (patientId: string) =>
  json<ApiDocument[]>(`/api/patients/${patientId}/documents`);

export async function uploadFile(file: File):
  Promise<{ staged_path: string; mime: string; ext: string }> {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${API}/api/chat/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json();
}

// Read an SSE stream from a POST response and dispatch to handlers.
async function readSse(res: Response, h: SseHandlers): Promise<void> {
  if (!res.ok || !res.body) {
    h.onError?.(`${res.status} ${res.statusText}`);
    h.onDone?.();
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const blocks = buf.split('\n\n');
    buf = blocks.pop() ?? '';
    for (const block of blocks) {
      let event = 'message';
      let data = '';
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) data += line.slice(5).trim();
      }
      let parsed: any = {};
      try { parsed = data ? JSON.parse(data) : {}; } catch { parsed = {}; }
      if (event === 'node') h.onNode?.(parsed.label);
      else if (event === 'progress') h.onProgress?.(parsed.msg);
      else if (event === 'interrupt') h.onInterrupt?.(parsed);
      else if (event === 'message') h.onMessage?.(parsed);
      else if (event === 'error') h.onError?.(parsed.message);
      else if (event === 'done') h.onDone?.();
    }
  }
}

export async function streamChat(body: {
  thread_id: string; message?: string; patient_id?: number | null;
  staged_path?: string; mime?: string; ext?: string;
}, handlers: SseHandlers): Promise<void> {
  const res = await fetch(`${API}/api/chat/stream`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  await readSse(res, handlers);
}

export async function resumeChat(body: { thread_id: string; resume: any },
                                 handlers: SseHandlers): Promise<void> {
  const res = await fetch(`${API}/api/chat/resume`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  await readSse(res, handlers);
}
```

- [ ] **Step 3: Edit `.env.example`** — append:

```
VITE_API_BASE=http://localhost:8000
```

- [ ] **Step 4: Type-check**

Run: `cd medagentic-dashboard && npm install && npm run lint`
Expected: `tsc --noEmit` exits 0 for api.ts + types.ts. `main.ts` still imports `./data` here — lint errors confined to `main.ts`/`data.ts` are expected and fixed in Task 9; api.ts/types.ts must be clean.

- [ ] **Step 5: Commit**

```bash
git add medagentic-dashboard/src/api.ts medagentic-dashboard/src/types.ts medagentic-dashboard/.env.example
git commit -m "feat(ui): typed API client with SSE reader"
```

---

## Task 9: Rewire `main.ts` to the API

**Files:**
- Modify: `medagentic-dashboard/src/main.ts` (full replacement)
- Delete: `medagentic-dashboard/src/data.ts`

Keep all existing markup/styling; only the data source and event wiring change. **Every
dynamic string rendered into `innerHTML` (patient/record/doc/chat text from the DB, OCR, or
LLM) is wrapped in `esc()` to neutralize XSS — the placeholder did not do this and it matters
for medical data.** Class strings and trusted constants are left as-is.

- [ ] **Step 1: Replace `src/main.ts`** with the API-driven version below.

```typescript
import './index.css';
import { ApiDocument, ApiPatient, ApiRecord } from './types';
import {
  createPatient, getDocuments, getHealth, getRecords, listPatients,
  resumeChat, streamChat, uploadFile,
} from './api';

declare const lucide: any;

// Escape untrusted text before it enters innerHTML (XSS guard for DB/OCR/LLM content).
function esc(v: unknown): string {
  return String(v ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

interface ChatMsg {
  sender: 'user' | 'agent';
  text: string;
  timestamp: string;
  sources?: string[];
  live?: boolean;       // agent bubble still streaming
  interrupt?: any;      // HITL payload -> render a card instead of a bubble
}

let patients: ApiPatient[] = [];
let records: ApiRecord[] = [];
let docs: ApiDocument[] = [];
let currentPatientId = '';
let filterType = 'all';
let sortOrder: 'desc' | 'asc' = 'desc';
let mobileTab: 'dashboard' | 'knowledge' = 'dashboard';
let panelTab: 'chat' | 'docs' = 'chat';
let chats: ChatMsg[] = [];
let stagedFileName = '';
const threadId = `web-${Math.random().toString(36).slice(2)}-${Date.now()}`;

const $ = (id: string) => document.getElementById(id);

function render() {
  renderSidebar();
  renderDashboard();
  renderChatbot();
  renderMobileTabs();
  if (typeof lucide !== 'undefined') lucide.createIcons();
}

function nowIso() { return new Date().toISOString(); }
function formatTime(s: string) {
  return new Date(s).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}
function formatDate(s: string) {
  return new Date(s).toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' });
}

async function init() {
  try {
    const h = await getHealth();
    if (h.db !== 'ok') banner(`Backend not ready: ${h.db}`);
  } catch (e: any) {
    banner(`Cannot reach API: ${e.message}`);
  }
  try {
    patients = await listPatients();
    if (patients.length) {
      currentPatientId = patients[0].id;
      await loadPatientData();
    }
  } catch (e: any) {
    banner(`Failed to load patients: ${e.message}`);
  }
  render();
}

function banner(msg: string) {
  let el = $('api-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'api-banner';
    el.className = 'fixed top-0 inset-x-0 z-50 bg-[#C16D54] text-white text-xs '
      + 'font-bold text-center py-2 px-4';
    document.body.prepend(el);
  }
  el.textContent = msg;  // textContent: safe by construction
}

async function loadPatientData() {
  if (!currentPatientId) { records = []; docs = []; return; }
  [records, docs] = await Promise.all([
    getRecords(currentPatientId).catch(() => []),
    getDocuments(currentPatientId).catch(() => []),
  ]);
}

async function selectPatient(id: string) {
  currentPatientId = id;
  filterType = 'all';
  await loadPatientData();
  render();
}

function renderSidebar() {
  const list = $('patient-list');
  if (list) {
    list.innerHTML = patients.map(p => `
      <button data-id="${esc(p.id)}" class="patient-btn w-full text-left px-3 py-3 rounded-xl transition-all flex items-start gap-4 border ${
        currentPatientId === p.id ? 'bg-[#2A2E2C] border-[#393E3A] shadow-sm' : 'border-transparent hover:bg-[#202322]'
      }">
        <img src="${esc(p.image)}" alt="${esc(p.name)}" class="w-10 h-10 rounded-full object-cover bg-[#2C302D] shrink-0" />
        <div class="flex-1 overflow-hidden">
          <div class="text-sm font-semibold truncate leading-none mt-1 ${currentPatientId === p.id ? 'text-[#F5F4F0]' : 'text-[#A2A5A0]'}">${esc(p.name)}</div>
          <div class="text-xs flex items-center gap-1 mt-1.5 ${currentPatientId === p.id ? 'text-[#878985]' : 'text-[#6D716E]'}">
             ${esc(p.age ?? '—')}y &bull; ${esc(p.gender ?? '—')} &bull; ${esc(p.bloodType)}
          </div>
        </div>
      </button>
    `).join('') + addPatientFormHtml();

    document.querySelectorAll('.patient-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        const id = (e.currentTarget as HTMLButtonElement).dataset.id;
        if (id) selectPatient(id);
      });
    });
    bindAddPatient();
  }

  const select = $('mobile-patient-select') as HTMLSelectElement;
  if (select) {
    select.innerHTML = patients.map(p =>
      `<option value="${esc(p.id)}" ${p.id === currentPatientId ? 'selected' : ''}>${esc(p.name)}</option>`).join('');
    select.onchange = e => selectPatient((e.target as HTMLSelectElement).value);
  }
}

function addPatientFormHtml() {
  return `
    <form id="add-patient-form" class="mt-4 pt-4 border-t border-[#232725] space-y-2">
      <input id="ap-name" placeholder="New patient name" class="w-full bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
      <div class="flex gap-2">
        <input id="ap-age" type="number" placeholder="Age" class="w-1/2 bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
        <input id="ap-gender" placeholder="Gender" class="w-1/2 bg-[#202322] text-[#F5F4F0] text-sm rounded-lg px-3 py-2 border border-[#2A2E2C] outline-none placeholder:text-[#6D716E]" />
      </div>
      <button type="submit" class="w-full bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white text-xs font-bold py-2 rounded-lg">Add patient</button>
    </form>`;
}

function bindAddPatient() {
  $('add-patient-form')?.addEventListener('submit', async e => {
    e.preventDefault();
    const name = ($('ap-name') as HTMLInputElement).value.trim();
    if (!name) return;
    const ageVal = ($('ap-age') as HTMLInputElement).value;
    const gender = ($('ap-gender') as HTMLInputElement).value.trim();
    const created = await createPatient({
      name, age: ageVal ? parseInt(ageVal, 10) : null, gender: gender || null });
    patients = await listPatients();
    await selectPatient(created.id);
  });
}

function renderDashboard() {
  const patient = patients.find(p => p.id === currentPatientId);
  const imgEl = $('header-patient-img') as HTMLImageElement;
  if (imgEl) imgEl.src = patient?.image ?? '';
  if ($('header-patient-name')) $('header-patient-name')!.innerText = patient?.name ?? '—';
  if ($('header-patient-meta')) {
    $('header-patient-meta')!.innerHTML = patient ? `
      <span>ID: ${esc(patient.id)}</span>
      <span class="w-1 h-1 bg-[#D9D7CF] rounded-full hidden sm:block"></span>
      <span class="hidden sm:block">${esc(patient.age ?? '—')} yrs</span>
      <span class="w-1 h-1 bg-[#D9D7CF] rounded-full"></span>
      <span>Blood: <strong class="text-[#2E2C29]">${esc(patient.bloodType)}</strong></span>` : '';
  }
  if ($('header-patient-date')) $('header-patient-date')!.innerText = patient?.lastVisit ?? '—';

  if ($('filter-buttons')) {
    const filters = ['all', 'disease', 'symptom', 'medicine', 'test_result', 'treatment_plan'];
    $('filter-buttons')!.innerHTML = filters.map(type => `
      <button data-type="${type}" class="filter-btn px-3 py-1.5 text-xs font-bold rounded-lg capitalize transition-all whitespace-nowrap ${
        filterType === type ? 'bg-white text-[#2E2C29] shadow-sm' : 'text-[#8C8982] hover:text-[#2E2C29]'
      }">${type === 'all' ? 'All' : type.replace('_', ' ')}</button>`).join('');
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', e => {
        filterType = (e.currentTarget as HTMLButtonElement).dataset.type!;
        renderDashboard();
        if (typeof lucide !== 'undefined') lucide.createIcons();
      });
    });
  }

  if ($('sort-label')) $('sort-label')!.innerText = sortOrder === 'desc' ? 'Newest' : 'Oldest';
  const oldBtn = $('sort-btn');
  if (oldBtn) {
    const newBtn = oldBtn.cloneNode(true);
    oldBtn.replaceWith(newBtn);
    newBtn.addEventListener('click', () => {
      sortOrder = sortOrder === 'desc' ? 'asc' : 'desc';
      renderDashboard();
      if (typeof lucide !== 'undefined') lucide.createIcons();
    });
  }

  const view = records
    .filter(r => filterType === 'all' || r.type === filterType)
    .sort((a, b) => {
      const da = new Date(a.date ?? 0).getTime();
      const db = new Date(b.date ?? 0).getTime();
      return sortOrder === 'desc' ? db - da : da - db;
    });

  const grid = $('records-grid');
  if (grid) {
    grid.innerHTML = view.length === 0 ? `
      <div class="col-span-full text-center py-16 md:py-20 text-[#A6A298]">
        <i data-lucide="filter" class="w-10 h-10 mx-auto text-[#D5D2C9] mb-4"></i>
        <p class="text-lg font-light tracking-tight">No records found for this filter.</p>
      </div>` : view.map(record => `
      <div class="bg-white rounded-3xl border border-[#E0DDD5] shadow-sm p-5 md:p-6 hover:shadow-md hover:-translate-y-1 transition-all flex flex-col">
        <div class="flex justify-between items-start mb-4">
          <div>
            <h3 class="text-[17px] font-bold text-[#2E2C29] leading-tight flex flex-col items-start gap-1.5">
              ${esc(record.title)}
              <span class="text-[9px] font-black text-[#A6A298] uppercase tracking-widest bg-[#F5F4F0] px-2 py-0.5 rounded-full">${esc(record.type.replace('_', ' '))}</span>
            </h3>
          </div>
          ${record.severity ? `
            <span class="px-2 py-1 text-[9px] font-bold rounded uppercase tracking-widest shadow-sm whitespace-nowrap mt-0.5 ${
              record.severity === 'High' || record.severity === 'Critical' ? 'bg-[#FF7373] text-white' : 'bg-[#E5B567] text-white'
            }">${esc(record.severity)}</span>` : `
            <span class="px-2 py-1 bg-[#F5F4F0] text-[#8C8982] text-[9px] font-bold rounded uppercase tracking-widest border border-[#EBEBE6] whitespace-nowrap mt-0.5">${esc(record.status)}</span>`}
        </div>
        <div class="flex-1 mt-1 mb-2">
          <p class="font-medium text-[#59554D] text-[13px] md:text-sm leading-relaxed">${esc(record.description)}</p>
        </div>
        <div class="mt-5 pt-4 border-t border-[#F0EFEB] flex flex-wrap gap-2 items-center justify-between">
          <p class="text-[10px] text-[#A6A298] font-bold uppercase tracking-wider flex items-center gap-1.5">
            <i data-lucide="calendar" class="w-3.5 h-3.5 text-[#5D7B6F]"></i> ${esc(record.date ?? '—')}
          </p>
          ${record.doctor ? `
            <p class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider flex items-center gap-1.5 bg-[#F5F4F0] px-2 py-1 rounded-md">
              <i data-lucide="stethoscope" class="w-3.5 h-3.5 text-[#5D7B6F]"></i> <span class="truncate max-w-[120px]">${esc(record.doctor)}</span>
            </p>` : ''}
        </div>
      </div>`).join('');
  }
}

function renderChatbot() {
  const tabChat = $('panel-tab-chat');
  const tabDocs = $('panel-tab-docs');
  const viewChat = $('view-chat');
  const viewDocs = $('view-docs');
  if (!tabChat || !tabDocs || !viewChat || !viewDocs) return;

  const activeCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all border-b-[3px] border-[#5D7B6F] text-[#2E2C29] bg-white';
  const idleCls = 'flex-1 flex items-center justify-center gap-2 py-4 px-4 text-[11px] md:text-xs font-bold uppercase tracking-widest transition-all border-b-[3px] border-[#EBEBE6] text-[#A6A298] hover:text-[#2E2C29] hover:bg-[#F5F4F0] bg-[#FAFAF8] shadow-inner';

  if (panelTab === 'chat') {
    tabChat.className = activeCls;
    tabChat.innerHTML = `<div class="w-2 h-2 rounded-full bg-[#5D7B6F]"></div> Agentic AI`;
    tabDocs.className = idleCls;
    tabDocs.innerHTML = `<i data-lucide="upload-cloud" class="w-3.5 h-3.5"></i> Knowledge`;
    viewChat.classList.remove('hidden'); viewChat.classList.add('flex');
    viewDocs.classList.add('hidden'); viewDocs.classList.remove('flex');
  } else {
    tabChat.className = idleCls;
    tabChat.innerHTML = `<div class="w-2 h-2 rounded-full bg-[#D5D2C9]"></div> Agentic AI`;
    tabDocs.className = activeCls;
    tabDocs.innerHTML = `<i data-lucide="upload-cloud" class="w-3.5 h-3.5"></i> Knowledge`;
    viewChat.classList.add('hidden'); viewChat.classList.remove('flex');
    viewDocs.classList.remove('hidden'); viewDocs.classList.add('flex');
  }

  renderMessages();
  renderDocs();
}

function renderMessages() {
  const el = $('chat-messages');
  if (!el) return;
  el.innerHTML = chats.map((msg, i) => {
    if (msg.interrupt) return interruptCardHtml(msg.interrupt, i);
    const bubble = msg.sender === 'user'
      ? 'bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white rounded-[20px] rounded-tr-[4px] shadow-sm'
      : 'bg-white border border-[#EBEBE6] text-[#2E2C29] rounded-[20px] rounded-tl-[4px] shadow-[0_2px_8px_rgba(0,0,0,0.02)]';
    return `
      <div class="flex flex-col gap-1.5 max-w-[90%] md:max-w-[85%] ${msg.sender === 'user' ? 'items-end ml-auto' : 'items-start'}">
        <div class="${bubble} p-3 md:p-4 text-[13px] leading-relaxed font-medium">
          ${esc(msg.text)}${msg.live ? ' <span class="animate-pulse">▍</span>' : ''}
          ${msg.sources && msg.sources.length ? `
            <div class="flex flex-wrap gap-2 mt-3 pt-3 border-t border-[#EBEBE6]/60">
              ${msg.sources.map(s => `
                <div class="flex items-center gap-1.5 py-1.5 px-2.5 bg-[#F5F4F0] border border-[#E0DDD5] rounded-xl text-[#2E2C29] shadow-sm">
                  <span class="text-[8px] md:text-[9px] font-bold text-[#5D7B6F] uppercase tracking-wider">[REF]</span>
                  <span class="text-[10px] md:text-[11px] font-bold truncate max-w-[150px]">${esc(s)}</span>
                </div>`).join('')}
            </div>` : ''}
        </div>
        <span class="text-[9px] md:text-[10px] text-[#A6A298] font-bold tracking-widest uppercase ${msg.sender === 'user' ? 'mr-2' : 'ml-2'}">
          ${msg.sender === 'user' ? 'You' : 'Agent'} &bull; ${esc(formatTime(msg.timestamp))}
        </span>
      </div>`;
  }).join('');
  el.scrollTop = el.scrollHeight;
  bindInterruptButtons();
}

function interruptCardHtml(payload: any, idx: number) {
  if (payload.type === 'confirm_ingest') {
    const ex = payload.extracted || {};
    const name = ex.patient_name || 'New patient';
    const summary: string[] = [];
    (ex.tests || []).forEach((t: any) =>
      summary.push(`${t.name}: ${t.value ?? ''}${t.unit ?? ''}`));
    ['diseases', 'symptoms', 'medications'].forEach(k =>
      (ex[k] || []).forEach((i: any) => summary.push(i.name)));
    return `
      <div class="bg-gradient-to-br from-[#F5F4F0] to-[#E9E8E1] rounded-3xl p-5 md:p-6 shadow-lg border border-[#DEDCD6]">
        <div class="flex items-center gap-2 mb-3 text-[#C16D54]">
          <i data-lucide="user" class="w-3.5 h-3.5"></i>
          <span class="font-extrabold text-[9px] md:text-[10px] tracking-widest uppercase">Human in the loop</span>
        </div>
        <h3 class="text-xl font-light text-[#2E2C29] mb-4 tracking-tight">Verify Extraction</h3>
        <div class="space-y-1.5 mb-5 bg-white/70 p-2 rounded-2xl border border-white">
          <div class="flex justify-between items-center p-2.5">
            <div class="text-[10px] font-bold text-[#8C8982] uppercase tracking-wider">Patient</div>
            <div class="text-sm font-bold text-[#2E2C29] bg-[#EBE9E4] px-2.5 py-1 rounded-md">${esc(name)}</div>
          </div>
          ${summary.slice(0, 6).map(s => `
            <div class="flex justify-between items-center p-2.5 text-[#59554D] text-xs font-semibold">${esc(s)}</div>`).join('')}
        </div>
        <div class="flex gap-2.5">
          <button data-act="reject" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] hover:text-[#C16D54] py-3 rounded-xl text-xs font-extrabold">Reject</button>
          <button data-act="confirm" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Confirm & Feed Layer</button>
        </div>
      </div>`;
  }
  // low_confidence
  return `
    <div class="bg-white rounded-3xl p-5 shadow-lg border border-[#DEDCD6]">
      <h3 class="text-lg font-light text-[#2E2C29] mb-2">Weak match (score ${esc(payload.score ?? '?')})</h3>
      <p class="text-xs text-[#8C8982] mb-4">Answer anyway from the records found, or skip?</p>
      <div class="flex gap-2.5">
        <button data-act="skip" data-idx="${idx}" class="int-btn flex-1 bg-white border border-[#DFDDDA] text-[#A6A298] py-3 rounded-xl text-xs font-extrabold">Skip</button>
        <button data-act="proceed" data-idx="${idx}" class="int-btn flex-[2] bg-gradient-to-br from-[#698A7D] to-[#4F6D61] text-white py-3 rounded-xl text-xs font-extrabold">Answer anyway</button>
      </div>
    </div>`;
}

function bindInterruptButtons() {
  document.querySelectorAll('.int-btn').forEach(btn => {
    btn.addEventListener('click', e => {
      const t = e.currentTarget as HTMLButtonElement;
      const idx = parseInt(t.dataset.idx!, 10);
      const payload = chats[idx]?.interrupt;
      if (!payload) return;
      chats.splice(idx, 1); // remove the card
      let resume: any;
      if (payload.type === 'confirm_ingest') {
        resume = t.dataset.act === 'confirm'
          ? { approved: true, extracted: payload.extracted,
              ...(payload.patient_id ? { patient_id: payload.patient_id } : {}) }
          : { approved: false };
      } else {
        resume = { proceed: t.dataset.act === 'proceed' };
      }
      runResume(resume);
    });
  });
}

function renderDocs() {
  const dz = $('upload-container');
  if (dz) {
    dz.innerHTML = `
      <div id="dropzone" class="border-2 border-dashed border-[#DFDDDA] rounded-3xl p-6 md:p-10 mb-8 text-center transition-all cursor-pointer group flex flex-col items-center justify-center min-h-[180px] md:min-h-[220px] bg-white hover:bg-[#FAF9F5]">
        <div class="w-14 h-14 md:w-16 md:h-16 bg-[#F5F4F0] rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300 shadow-inner">
          <i data-lucide="upload-cloud" class="w-6 h-6 md:w-7 md:h-7 text-[#8C8982] group-hover:text-[#5D7B6F] transition-colors"></i>
        </div>
        <div>
          <h3 class="text-lg md:text-xl font-light tracking-tight text-[#2E2C29] mb-1.5">Feed Knowledge Base</h3>
          <p class="text-[11px] md:text-sm font-medium mt-1 text-[#8C8982]">${esc(stagedFileName || 'Click to upload a PDF or image')}</p>
        </div>
        <input id="file-input" type="file" accept=".png,.jpg,.jpeg,.webp,.pdf,.txt" class="hidden" />
      </div>`;
    const zone = $('dropzone');
    const input = $('file-input') as HTMLInputElement;
    zone?.addEventListener('click', () => input?.click());
    input?.addEventListener('change', () => {
      if (input.files && input.files[0]) handleUpload(input.files[0]);
    });
    zone?.addEventListener('dragover', e => { e.preventDefault();
      zone.classList.add('border-[#5D7B6F]'); });
    zone?.addEventListener('drop', e => {
      e.preventDefault();
      const f = (e as DragEvent).dataTransfer?.files?.[0];
      if (f) handleUpload(f);
    });
  }

  const docsList = $('docs-list');
  if (docsList) {
    docsList.innerHTML = docs.map(doc => `
      <div class="bg-white border border-[#EBEBE6] p-3.5 md:p-4 rounded-2xl flex items-start gap-3 md:gap-4 shadow-sm">
        <div class="bg-[#FAF9F5] text-[#C16D54] p-3 rounded-xl shrink-0 hidden sm:block">
          <i data-lucide="file-text" class="w-5 h-5"></i>
        </div>
        <div class="flex-1 overflow-hidden pt-0.5">
          <div class="text-[13px] md:text-sm font-bold text-[#2E2C29] truncate tracking-tight" title="${esc(doc.name)}">${esc(doc.name)}</div>
          <div class="flex flex-wrap items-center gap-1.5 mt-1.5 text-[10px] md:text-[11px] text-[#A6A298] font-bold uppercase tracking-wider">
            <span>${esc(doc.type)}</span>
            <span class="w-1 h-1 rounded-full bg-[#D5D2C9]"></span>
            <span>${esc(doc.size)}</span>
          </div>
          ${doc.date ? `
          <div class="mt-2.5 text-[9px] md:text-[10px] font-bold text-[#8C8982] uppercase tracking-wider flex items-center gap-1.5 bg-[#FAF9F5] inline-flex px-2 py-1 rounded-md border border-[#EBEBE6]">
            <i data-lucide="upload-cloud" class="w-3 h-3"></i> ${esc(formatDate(doc.date))}
          </div>` : ''}
        </div>
      </div>`).join('');
  }
}

function renderMobileTabs() {
  const tDash = $('tab-dashboard'); const tKnow = $('tab-knowledge');
  const dashView = $('dashboard-view'); const knowView = $('knowledge-view');
  if (!tDash || !tKnow || !dashView || !knowView) return;
  const on = 'px-3 py-1.5 text-[11px] md:text-xs font-bold rounded-md transition-all bg-white shadow-sm text-[#2E2C29]';
  const off = 'px-3 py-1.5 text-[11px] md:text-xs font-bold rounded-md transition-all text-[#8C8982]';
  if (mobileTab === 'dashboard') {
    tDash.className = on; tKnow.className = off;
    dashView.classList.remove('hidden'); dashView.classList.add('flex');
    knowView.classList.remove('flex'); knowView.classList.add('hidden', 'lg:flex');
  } else {
    tDash.className = off; tKnow.className = on;
    dashView.classList.add('hidden'); dashView.classList.remove('flex');
    knowView.classList.add('flex'); knowView.classList.remove('hidden', 'lg:flex');
  }
}

// ---- chat actions ----

function liveAgent(): ChatMsg {
  const m: ChatMsg = { sender: 'agent', text: '…', timestamp: nowIso(), live: true };
  chats.push(m);
  return m;
}

function streamHandlers(agent: ChatMsg) {
  return {
    onNode: (label: string) => { agent.text = label; renderMessages(); },
    onProgress: (msg: string) => { agent.text = msg; renderMessages(); },
    onInterrupt: (payload: any) => {
      const i = chats.indexOf(agent);
      if (i >= 0) chats.splice(i, 1);
      chats.push({ sender: 'agent', text: '', timestamp: nowIso(), interrupt: payload });
      render();
    },
    onMessage: (m: { content: string; sources?: string[] }) => {
      agent.text = m.content; agent.live = false; agent.sources = m.sources;
      renderMessages();
    },
    onError: (message: string) => {
      agent.text = `⚠️ ${message}`; agent.live = false; renderMessages();
    },
    onDone: async () => {
      agent.live = false;
      await loadPatientData();   // ingest may have added records/docs
      render();
    },
  };
}

async function handleUpload(file: File) {
  panelTab = 'chat';
  stagedFileName = file.name;
  chats.push({ sender: 'user', text: `📎 ${file.name}`, timestamp: nowIso() });
  const agent = liveAgent();
  render();
  try {
    const staged = await uploadFile(file);
    await streamChat({ thread_id: threadId, message: 'Read this and arrange it.',
      staged_path: staged.staged_path, mime: staged.mime, ext: staged.ext },
      streamHandlers(agent));
  } catch (e: any) {
    agent.text = `⚠️ ${e.message}`; agent.live = false; renderMessages();
  } finally {
    stagedFileName = '';
  }
}

async function runResume(resume: any) {
  const agent = liveAgent();
  render();
  await resumeChat({ thread_id: threadId, resume }, streamHandlers(agent));
}

function sendText(text: string) {
  chats.push({ sender: 'user', text, timestamp: nowIso() });
  const agent = liveAgent();
  render();
  streamChat({ thread_id: threadId, message: text,
    patient_id: currentPatientId ? parseInt(currentPatientId, 10) : null },
    streamHandlers(agent));
}

// ---- global listeners (delegated; markup is re-rendered) ----
document.addEventListener('click', e => {
  const id = (e.target as HTMLElement).closest('button')?.id;
  if (id === 'panel-tab-chat') { panelTab = 'chat'; render(); }
  else if (id === 'panel-tab-docs') { panelTab = 'docs'; render(); }
  else if (id === 'tab-dashboard') { mobileTab = 'dashboard'; render(); }
  else if (id === 'tab-knowledge') { mobileTab = 'knowledge'; render(); }
});

$('chat-form')?.addEventListener('submit', e => {
  e.preventDefault();
  const input = $('chat-input') as HTMLInputElement;
  const val = input.value.trim();
  if (!val) return;
  input.value = '';
  ($('send-btn') as HTMLButtonElement).disabled = true;
  sendText(val);
});

init();
```

- [ ] **Step 2: Delete the mock data file**

Run: `git rm medagentic-dashboard/src/data.ts`
Expected: `rm 'medagentic-dashboard/src/data.ts'`

- [ ] **Step 3: Type-check**

Run: `cd medagentic-dashboard && npm run lint`
Expected: `tsc --noEmit` exits 0 — no references to `./data`, no type errors.

- [ ] **Step 4: Commit**

```bash
git add medagentic-dashboard/src/main.ts
git rm medagentic-dashboard/src/data.ts
git commit -m "feat(ui): drive dashboard from the API (streaming chat + HITL cards, esc XSS guard)"
```

---

## Task 10: Full-test run + manual smoke test + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the whole backend suite**

Run: `pytest -q`
Expected: all tests pass (existing suite plus the new `test_api_*`). If a pre-existing test was
already failing before this work, note it but do not block on it.

- [ ] **Step 2: Start backend + frontend, smoke test the integration**

Terminal A: `uvicorn app.api.server:app --port 8000 --workers 1`
Terminal B: `cd medagentic-dashboard && npm run dev` (serves on :3000)

In the browser at `http://localhost:3000`:
1. Sidebar lists patients from the DB (or empty). Add a patient via the sidebar form -> it appears.
2. Switch patients -> records grid + Knowledge docs reload for that patient.
3. Knowledge tab -> upload a PDF/image -> live progress labels stream in chat -> a "Verify
   Extraction" card appears -> Confirm -> records/documents refresh.
4. Chat tab with a patient selected -> ask a question -> node labels stream -> final answer with
   `[REF]` sources. Without a patient selected -> red error bubble asking to pick a patient.

Expected: each step behaves as described; no console errors blocking the flow.

- [ ] **Step 3: Update `README.md`** — replace the Streamlit run instruction with the API + UI commands. Add (or replace the existing Run section with):

```markdown
## Run

Backend (FastAPI, single worker — the agent checkpointer is in-process memory):

    uvicorn app.api.server:app --port 8000 --workers 1

Frontend (Vite dev server on :3000):

    cd medagentic-dashboard
    npm install
    npm run dev

Set `VITE_API_BASE` in `medagentic-dashboard/.env` if the API is not on `http://localhost:8000`.
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: run instructions for FastAPI backend + Vite UI"
```

---

## Self-Review notes (reconciled against the spec)

- **Spec coverage:** health/patients/records/documents (Task 6); upload/stream/resume SSE + HITL
  (Tasks 5,7,9); mapping incl. empty/derived fields (Task 3); Streamlit removal (Task 1); frontend
  client + rewire (Tasks 8,9); single-worker + run docs (Task 10). `treatment_plan` chip present
  but empty (Task 9 filter list; `merge_records` emits none).
- **Type consistency:** SSE event names (`node/progress/interrupt/message/error/done`) identical
  across `sse.py`, `api.ts`, `main.ts`. `streamChat`/`resumeChat`/`uploadFile`/`getRecords`/
  `createPatient` names match between `api.ts` and `main.ts`. `merge_records` signature identical
  in tests and impl.
- **Security:** `esc()` wraps all DB/OCR/LLM-derived strings in `main.ts` (XSS guard the
  placeholder lacked); `banner()` uses `textContent`.
- **Known constraint:** `MemorySaver` requires `--workers 1`; documented in Task 10 + spec.
- **Cross-task ordering caveat:** `server.py` (Task 6) imports `routes_chat` (Task 7); Task 6
  Step 4-5 notes the temporary comment-out for isolated testing, restored in Task 7 Step 4.
