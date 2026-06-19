# MedAgentic UI ↔ Backend Integration — Design

**Date:** 2026-06-17
**Branch:** feat/agentic_chatbot
**Goal:** Replace Streamlit with the `medagentic-dashboard` placeholder UI, wired to the existing
Python service + LangGraph agent layer over an HTTP API. Preserve live node-progress streaming and
human-in-the-loop (HITL) approval gates.

## Context

- **Backend** (`app/`): SQLAlchemy models on Supabase + pgvector, a tested service layer
  (`patients`, `documents`, `browse`, `health`, …), and a LangGraph supervisor
  (`app/agent/graph.py`) that routes to ingest / structured-query / RAG subgraphs with two HITL
  interrupts (`confirm_ingest`, `low_confidence`). Streaming + interrupts currently driven only by
  `streamlit_app.py` calling the graph in-process with a `MemorySaver` checkpointer.
- **Frontend** (`medagentic-dashboard/`): vanilla TypeScript + Vite + Tailwind (despite the
  `react-example` package name and unused react deps). `src/main.ts` renders the whole UI from
  mock `src/data.ts`. One patient-centric screen: patient cohort sidebar (swappable), per-patient
  Clinical Records grid with type filter chips, a chat panel, and a Knowledge/upload panel.
- **No HTTP layer exists today.** Streamlit is the only driver.

## Decisions (locked with user)

1. **Full fidelity agent transport:** SSE live node progress + HITL gates + resume. Not auto-approve.
2. **Keep vanilla TS** placeholder; add an API client, do not rewrite to React.
3. **Keep the original UI layout.** Patient-swap works. Fold all the old Streamlit browse data
   (diseases/symptoms/meds/diagnostics/documents) into the per-patient Records grid.
4. **`treatment_plan` filter chip stays but renders empty** (no backend table yet).

## Architecture

Approach A: a FastAPI sidecar wrapping the existing services + the LangGraph graph. The graph is
compiled **once at startup** with an in-process `MemorySaver`, keyed by `thread_id` — mirroring
`streamlit_app._graph` / `_deps` / `_cfg`. Single uvicorn worker (documented constraint, since the
checkpointer is in-process memory).

### Backend package `app/api/`

| File | Purpose |
|------|---------|
| `__init__.py` | package marker |
| `server.py` | FastAPI app, CORS (allow Vite origin), mount routers |
| `runtime.py` | singletons: `get_graph()` (build once), `get_deps()` (build once, probes embedder), `cfg(thread_id, progress=None)` builder |
| `schemas.py` | Pydantic response models: `HealthOut`, `PatientOut`, `PatientIn`, `RecordOut`, `DocumentOut` |
| `mapping.py` | pure functions: DB service rows → UI-shaped dicts |
| `routes_browse.py` | health, patients (list/create), records, documents |
| `routes_chat.py` | SSE stream / upload / resume |

**Run:** `uvicorn app.api.server:app --port 8000 --workers 1`

### Endpoints

```
GET  /api/health                      -> HealthOut {db, pgvector, version}
GET  /api/patients                    -> [PatientOut]
POST /api/patients                    -> PatientOut         body PatientIn {name, age?, gender?, relationship?}
GET  /api/patients/{id}/records       -> [RecordOut]        diseases + symptoms + meds + test_results merged
GET  /api/patients/{id}/documents     -> [DocumentOut]
POST /api/chat/upload                 -> {staged_path, mime, ext}    multipart, sync, stages bytes via app.storage.save_staging
POST /api/chat/stream  (SSE)          body {thread_id, message?, patient_id?, staged_path?, mime?, ext?}
POST /api/chat/resume  (SSE)          body {thread_id, resume:{...}}
```

`/chat/stream` builds the same state payloads `streamlit_app.chat_page` builds:
- **with `staged_path`** → ingest state (`file_path`, `mime_type`, `file_ext`, `source_type`,
  `messages`) and auto-starts ingest.
- **text only** → `{messages, patient_id?}`; if no `patient_id`, emit an `error` event telling the
  user to pick a patient (mirrors the Streamlit warning).

### SSE event schema

Server-Sent Events; frontend parses via `fetch` + `ReadableStream` reader (EventSource is GET-only,
and these are POSTs).

```
event:node      data:{label}              # from _NODE_LABELS map (moved into runtime.py), e.g. "📖 Reading the document (OCR)…"
event:progress  data:{msg}                # long-node sub-line (per-page OCR), via cfg progress callback
event:interrupt data:{type, ...payload}   # type ∈ {confirm_ingest, low_confidence}; full interrupt value passed through
event:message   data:{role, content, sources?}   # final assistant message from graph state
event:error     data:{message}            # provider/OCR/setup failure
event:done      data:{}                   # stream complete (terminal or paused-at-interrupt)
```

Driver logic ports `streamlit_app._drive`: iterate `graph.stream(payload, cfg, stream_mode="updates")`,
translate each node key to a `node` event, capture `__interrupt__` → `interrupt` event, then read
`graph.get_state(cfg)` for the final `messages` → `message` event. The `cfg` progress callback emits
`progress` events. On exception → `error` event. Always end with `done`.

`/chat/resume` calls the same driver with `Command(resume=<body.resume>)`.

## Data mapping (DB → placeholder UI shapes)

The placeholder `types.ts` expects fields the schema lacks. Derive; never fabricate clinical values.

**Patient** (`PatientOut`): `id` (int→str), `name`, `age`, `gender` from DB.
- `bloodType` → `"—"` (not in schema).
- `image` → `https://i.pravatar.cc/150?u=<url-encoded name>`.
- `lastVisit` → latest `document.uploaded_at` for that patient (via `documents.count`/timeline query), else `created_at` date.
- `status` → `"Active"` default.

**Record** (`RecordOut`): merge four sources for a patient:
- `browse.list_entity_links(db, "disease")` → `type:"disease"`
- `browse.list_entity_links(db, "symptom")` → `type:"symptom"`
- `browse.list_entity_links(db, "medication")` → `type:"medicine"`
- `browse.list_test_results(db)` → `type:"test_result"`, `title = f"{test}: {value}{unit}"`

Each record: `id` (synthetic `f"{type}-{document_id}-{i}"`), `patientId` (str), `type`, `title`
(entity name), `description` (`source_span` or doc_type), `date` (doc date), `status` (`"Recorded"`),
`severity` (omitted — falls back to the status pill in the card), `doctor` (omitted; backend doesn't
join doctor per entity yet). `treatment_plan` produces no rows.

**Document** (`DocumentOut`): from `browse.list_documents_timeline(db, patient_id)`. `name` →
basename of `file_path` (or `f"document-{id}.{ext}"`), `type` → `doc_type` or extension upper-cased,
`size` → file size from disk if readable else `"—"`, `date` → uploaded_at ISO.

## Frontend changes (`medagentic-dashboard/src/`)

- **`api.ts`** (new): typed client. `const API = import.meta.env.VITE_API_BASE ?? "http://localhost:8000"`.
  Functions: `getHealth`, `listPatients`, `createPatient`, `getRecords(patientId)`,
  `getDocuments(patientId)`, `uploadFile(file)`, `streamChat(body, handlers)`,
  `resumeChat(body, handlers)`. `handlers = {onNode, onProgress, onInterrupt, onMessage, onError, onDone}`.
  SSE parser reads the `fetch` response body stream, splits on `\n\n`, dispatches by `event:`/`data:`.
- **`main.ts`** (rewire, keep all existing markup/styling):
  - State arrays `patients`, `records`, `docs` start empty; `init()` awaits `listPatients()`, sets
    `currentPatientId`, then `render()`.
  - Selecting a patient (sidebar button / mobile select) re-fetches `getRecords` + `getDocuments`
    for that id, then re-renders.
  - Chat submit → push user bubble → push a live agent bubble → `streamChat`:
    `onNode`/`onProgress` update the live bubble text; `onMessage` finalizes it (with `sources`);
    `onInterrupt` renders an approval card; `onError` renders a red bubble.
  - `confirm_ingest` interrupt → reuse the existing `uploadState==="review"` markup (Verify
    Extraction card) populated from the interrupt payload (patient name, extracted summary,
    confidence); Confirm → `resumeChat({approved:true,...})`, Reject → `resumeChat({approved:false})`.
  - `low_confidence` interrupt → inline "Answer anyway / Skip" card in chat → `resumeChat({proceed})`.
  - Knowledge dropzone → real `uploadFile(file)` → `streamChat` with staged file; replace the fake
    `setInterval` progress bar with real `progress`/`node` events (keep the same visual component).
  - Health banner if `getHealth().db !== "ok"`.
  - A persistent `thread_id` (uuid) generated once per page load, sent on every stream/resume.
- **`data.ts`** → deleted. **`types.ts`** → kept; add API response types.
- **`.env.example`** → add `VITE_API_BASE`.

## Remove Streamlit

- Delete `streamlit_app.py`.
- Remove `streamlit==1.41.1` from `requirements.txt`.
- Add `fastapi`, `uvicorn[standard]`, `python-multipart`, `sse-starlette`.

## Testing (TDD)

- **`mapping.py`** — pure unit tests: rows → UI shapes, including empty input, missing optional
  fields, test-result title formatting, pravatar URL encoding. No DB.
- **Browse routes** — FastAPI `TestClient` against the live Supabase DB (read-only), matching the
  existing live-DB test convention. Assert shapes + status codes; create+read a throwaway patient.
- **Chat routes** — inject a fake graph (stub `.stream` yielding scripted update chunks incl.
  `__interrupt__`, and `.get_state` returning messages) via `runtime.get_graph` override. Assert the
  SSE event sequence for: clean RAG answer, ingest interrupt → resume, error path. No real OCR/LLM.

## Error handling

- Health endpoint gates the UI (banner on `db != ok`).
- Chat provider/OCR/setup failures → `event:error` → red bubble (mirrors `st.error`).
- Unknown/stale interrupt `type` → frontend dismiss card (mirrors Streamlit self-heal branch).
- Upload of unsupported extension → 400 from `/chat/upload`.

## Out of scope

- `treatment_plan` and per-entity `doctor`/`severity` (no backend tables) — chips/fields remain,
  render empty/omitted.
- Multi-worker deployment (in-process checkpointer requires one worker; Redis/DB checkpointer is a
  later concern).
- Document delete (UI has an `x` button; hide it for v1 — no delete endpoint).
