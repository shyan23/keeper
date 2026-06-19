# Agentic Chatbot — Design

Date: 2026-06-16
Branch: `feat/agentic_chatbot`
Status: Approved (design). Builds on Foundation (complete).

## Vision

A LangGraph agent that, in one conversational surface, does the work of the old
plan's sub-projects 2 (ingestion/extraction), 3 (RAG), and 5 (human-in-the-loop):

- "Read this file and arrange it" → OCR/extract → parse patient name, symptoms,
  meds, tests → **you confirm/edit** → persist + index for RAG.
- "Show me the latest medical report of patient X" → structured DB lookup.
- "What did the report say about Y?" → grounded RAG answer **with proof**
  (every claim cites a chunk id + source span).
- **Human in the loop always** — four approval gates.

## Locked decisions

| Layer | Choice |
|---|---|
| Orchestration | **LangGraph** single `StateGraph`: supervisor router → 3 subgraphs (ingest / structured_query / rag_query). |
| Chat + vision LLM | **Groq** (`llama-3.3-70b-versatile` for chat/extraction; a Groq **vision** model, e.g. `meta-llama/llama-4-scout-17b-16e-instruct`, for image OCR). |
| Embeddings | **Ollama `nomic-embed-text`** — local, free, 768-dim → matches existing `chunk.embedding vector(768)`. |
| Text PDF | `pypdf` direct text extraction (no OCR when a text layer exists). |
| Reranking | **LLM-based grading via Groq** (self-RAG style). No cross-encoder / torch. |
| UI | Existing **Streamlit** app — add a "Chat" page. |
| HITL | LangGraph `interrupt()`; `MemorySaver` checkpointer held in `st.session_state`. |
| Isolation | Patient-scoped: `WHERE patient_id = ?` on retrieval (schema already denormalizes `patient_id` onto `chunk`). |

> Provider note: spec's old Gemini plan is superseded. Gemini has no role here.
> Groq has no embeddings API → Ollama supplies embeddings.

## Architecture

### Graph topology

```
                 ┌──────────┐
   user turn  →  │  router  │   Groq classifies intent
                 └────┬─────┘
        ┌─────────────┼──────────────┐
        ▼             ▼              ▼
   [ingest]   [structured_query]  [rag_query]
   subgraph       subgraph         subgraph
```

### State (`app/agent/state.py`)

`AgentState` (TypedDict):

- `messages: list` — conversation
- `intent: str | None` — `ingest | structured_query | rag_query`
- Ingest: `document_id`, `file_path`, `ocr_text`, `extracted` (Pydantic
  `ExtractionResult`), `patient_id`, `patient_candidates`
- Query: `query_filters: dict | None`, `retrieved: list`, `answer: str | None`,
  `citations: list`

Pydantic schemas (same file): `ExtractionResult` with `patient` (name/age/gender),
`doctor`, `diseases[]`, `symptoms[]`, `medications[]`, `tests[]` (name/value/unit/
reference_range), `doc_type`, `doc_date`. Every field group carries `confidence:
float` and `source_span: str` (the proof surfaced at the HITL gate).

### Ingest subgraph (`app/agent/nodes/ingest.py`)

```
load_document → extract_text → extract_entities → ⟨HITL confirm+edit⟩
   → resolve_patient → ⟨HITL confirm patient if ambiguous⟩
   → persist → chunk_and_embed → done
```

- **load_document**: resolve `document_id`/`file_path` from upload (reuses
  `app/services/documents.py` + `app/storage.py`).
- **extract_text** (`app/services/extraction.py`): text-PDF → `pypdf`; image or
  image-PDF → Groq vision OCR. Result written to `document.raw_ocr_text`,
  `document.status='ocr_done'`.
- **extract_entities**: Groq `with_structured_output(ExtractionResult)` over
  `ocr_text`.
- **resolve_patient**: match extracted name against `patient` table (case-insensitive).
  Exact single match → use it. Zero/multiple → set `patient_candidates`, mark ambiguous.
- **persist** (`app/services/entities.py`): upsert into normalized entity tables
  (doctor/disease/symptom/medication/medical_test/test_result), create
  `document_entity` links with `confidence`, `source_span`, `validated=True`.
- **chunk_and_embed** (`app/services/chunking.py`): contextual-header chunking
  (prefix each chunk with `patient · doc_type · date`), embed via Ollama, write
  `chunk` rows (incl. denormalized `patient_id`). `document.status='indexed'`.

### structured_query subgraph (`app/agent/nodes/structured.py`)

```
parse_filters(Groq) → query_db → format_answer
```

Parses patient name + doc_type + date-ordering from the turn; calls
`app/services/documents.py` (filter + `ORDER BY uploaded_at DESC LIMIT n`).
No vector search — correct tool for "latest/newest/last".

### rag_query subgraph (`app/agent/nodes/rag.py`)

```
embed_query(Ollama) → retrieve(pgvector, patient-scoped)
   → grade(Groq) → ⟨HITL if low confidence⟩ → generate_grounded_answer
```

- **retrieve** (`app/services/retrieval.py`): pgvector cosine top-k with
  `WHERE patient_id = ?`.
- **grade**: Groq scores retrieved chunks' relevance (0–1). Mean below threshold
  → low-confidence interrupt.
- **generate_grounded_answer**: answer strictly from retrieved chunks; each claim
  cites `[chunk #id, "source span", doc_type, date]`. Refuse if nothing relevant.

## HITL gates (`interrupt()`)

| Gate | Node | Resume payload |
|---|---|---|
| Confirm + **edit** extracted entities (also = "approve DB write") | after `extract_entities` | edited entity dict + approve/reject |
| Confirm patient match / create-new | `resolve_patient` (only if ambiguous) | chosen `patient_id` or "new" |
| Confirm low-confidence answer | rag `grade` < threshold | proceed / refine |

"Approve any DB write" is folded into the entity-confirm gate: the review→edit→commit
step is the same data moment, so one approval commits exactly what you saw. No other
node writes without passing this gate. Streamlit renders the widget; on submit the
graph resumes from the checkpoint (OCR/extraction never re-run).

## Module layout

```
app/agent/
  state.py          # AgentState + Pydantic extraction schemas
  graph.py          # build_graph(): supervisor wiring + checkpointer
  router.py         # intent classification node
  llm.py            # Groq chat + vision client factory
  embeddings.py     # Ollama nomic-embed-text client
  nodes/
    ingest.py
    structured.py
    rag.py
app/services/
  extraction.py     # pypdf + Groq vision OCR
  entities.py       # upsert entities + document_entity links
  retrieval.py      # pgvector patient-scoped similarity search
  chunking.py       # chunk + embed + persist chunks
streamlit_app.py    # + Chat page driving the graph, rendering interrupts
```

## Config additions (`app/config.py`)

- `groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"`
- `ollama_embed_model: str = "nomic-embed-text"`
- `rag_top_k: int = 5`
- `rag_confidence_threshold: float = 0.5`

## Dependencies (add to `requirements.txt`)

`langgraph`, `langchain-core`, `langchain-groq`, `langchain-ollama`, `pypdf`.

## Testing (TDD)

- Each node unit-tested with **injected fake LLM/embedder** (no network) — assert
  state transitions, extraction shape, citation formatting, patient-scope filter.
- Graph integration test with stubbed Groq/Ollama covering: ingest happy path,
  ambiguous-patient gate, structured "latest report" query, RAG answer with
  citations, low-confidence interrupt + resume.
- DB-touching tests against the Supabase pooler DB (as existing suite).

## Out of scope (this branch)

Multi-agent (single supervisor only), LangSmith tracing, automated eval harness,
auth, deployment, cross-time timeline/organization layer (old sub-project 6).

## Success criteria

- Upload a prescription image → agent OCRs, extracts patient/symptoms/meds/tests →
  HITL confirm/edit → data persisted + chunks embedded.
- "Latest report of <patient>" returns the correct most-recent document.
- A content question returns a grounded answer citing chunk ids + source spans,
  scoped to that patient only.
- Low-confidence retrieval pauses for confirmation; resume continues without
  re-running upstream nodes.
- All nodes unit-tested headless; graph integration test green.
