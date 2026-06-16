# Agentic Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A LangGraph supervisor agent (router → ingest / structured_query / rag_query subgraphs) that ingests medical files (OCR + entity extraction), answers structured and grounded-RAG questions with citations, and pauses for human approval at four gates — surfaced in the existing Streamlit app.

**Architecture:** One `StateGraph` over a typed `AgentState`. A Groq-backed router classifies each turn and dispatches to a subgraph. Ingest does pypdf/Groq-vision OCR → Groq structured extraction → HITL confirm → persist + Ollama-embedded chunks. RAG retrieves patient-scoped pgvector chunks, grades with Groq, and answers with `[chunk #id, "span"]` citations. HITL uses LangGraph `interrupt()` with a `MemorySaver` checkpointer.

**Tech Stack:** Python, LangGraph, langchain-groq (chat+vision), langchain-ollama (`nomic-embed-text`, 768-dim), pypdf, SQLAlchemy 2 + pgvector (Supabase pooler), Streamlit, pytest.

---

## Dependency injection convention

Nodes are `def node(state, config)` and read clients from `config["configurable"]["deps"]` — a `Deps` dataclass (Task 2). Services take explicit client args (never reach into config). Tests pass fakes; production passes real Groq/Ollama clients. This keeps every node and service testable headless with no network.

Protocols (structural, defined in Task 2):
- `Embedder`: `embed_query(text: str) -> list[float]`, `embed_documents(texts: list[str]) -> list[list[float]]` (matches `OllamaEmbeddings`).
- `ChatLLM`: `complete(prompt: str) -> str`, `structured(prompt: str, schema: type[BaseModel]) -> BaseModel`.
- `VisionLLM`: `ocr_image(data: bytes, mime: str) -> str`.

---

## File structure

```
app/agent/
  state.py          # AgentState TypedDict, Pydantic extraction schemas, Deps, protocols
  embeddings.py     # OllamaEmbedder (wraps OllamaEmbeddings)
  llm.py            # GroqChat, GroqVision client factories
  router.py         # classify_intent node
  graph.py          # build_graph(): wiring + checkpointer
  nodes/
    __init__.py
    ingest.py
    structured.py
    rag.py
app/services/
  extraction.py     # extract_text: pypdf | vision OCR
  entities.py       # persist_extraction: upsert entities + document_entity links
  chunking.py       # chunk_and_embed: header chunks + embed + persist
  retrieval.py      # search_chunks: pgvector cosine, patient-scoped
streamlit_app.py    # + Chat page
app/config.py       # + groq_vision_model, ollama_embed_model, rag_top_k, rag_confidence_threshold
requirements.txt    # + langgraph, langchain-core, langchain-groq, langchain-ollama, pypdf
```

---

## Task 1: Dependencies + config

**Files:**
- Modify: `requirements.txt`
- Modify: `app/config.py:9-19`
- Test: `tests/test_config.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_config.py`:

```python
def test_agent_config_defaults(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://x")
    from app.config import Settings
    s = Settings()
    assert s.groq_vision_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert s.ollama_embed_model == "nomic-embed-text"
    assert s.rag_top_k == 5
    assert s.rag_confidence_threshold == 0.5
```

- [ ] **Step 2: Run it, verify fail**

Run: `pytest tests/test_config.py::test_agent_config_defaults -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'groq_vision_model'`

- [ ] **Step 3: Add config fields**

In `app/config.py`, after `ollama_model` line, add:

```python
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    ollama_embed_model: str = "nomic-embed-text"
    rag_top_k: int = 5
    rag_confidence_threshold: float = 0.5
```

- [ ] **Step 4: Add deps to `requirements.txt`**

Append:

```
langgraph==0.2.60
langchain-core==0.3.28
langchain-groq==0.2.2
langchain-ollama==0.2.2
pypdf==5.1.0
```

- [ ] **Step 5: Install + run test**

Run: `pip install -r requirements.txt && pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt app/config.py tests/test_config.py
git commit -m "feat: agent config fields + langgraph/groq/ollama deps"
```

---

## Task 2: State, schemas, protocols, Deps

**Files:**
- Create: `app/agent/__init__.py` (empty)
- Create: `app/agent/state.py`
- Test: `tests/agent/test_state.py`
- Create: `tests/agent/__init__.py` (empty)

- [ ] **Step 1: Write the failing test**

`tests/agent/test_state.py`:

```python
from app.agent.state import ExtractionResult, ExtractedEntity, AgentState, Deps


def test_extraction_result_roundtrip():
    er = ExtractionResult(
        patient_name="Jane Doe", patient_age=40, patient_gender="F",
        doc_type="prescription", doc_date="2026-06-10",
        doctor="Dr. Smith",
        diseases=[ExtractedEntity(name="hypertension", confidence=0.9, source_span="Dx: HTN")],
        symptoms=[], medications=[], tests=[],
        confidence=0.8, source_span="full doc",
    )
    assert er.patient_name == "Jane Doe"
    assert er.diseases[0].confidence == 0.9
    # JSON-serializable (LLM structured output target)
    assert "hypertension" in er.model_dump_json()


def test_agent_state_is_typeddict():
    st: AgentState = {"messages": [], "intent": None}
    assert st["intent"] is None
```

- [ ] **Step 2: Run it, verify fail**

Run: `pytest tests/agent/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agent.state'`

- [ ] **Step 3: Implement**

`app/agent/state.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field


# ---- Extraction schemas (Groq structured-output target) ----

class ExtractedEntity(BaseModel):
    name: str
    confidence: float = 0.5
    source_span: str = ""


class ExtractedTest(BaseModel):
    name: str
    value: str | None = None
    unit: str | None = None
    reference_range: str | None = None
    confidence: float = 0.5
    source_span: str = ""


class ExtractionResult(BaseModel):
    patient_name: str | None = None
    patient_age: int | None = None
    patient_gender: str | None = None
    doc_type: str | None = None
    doc_date: str | None = None
    doctor: str | None = None
    diseases: list[ExtractedEntity] = Field(default_factory=list)
    symptoms: list[ExtractedEntity] = Field(default_factory=list)
    medications: list[ExtractedEntity] = Field(default_factory=list)
    tests: list[ExtractedTest] = Field(default_factory=list)
    confidence: float = 0.5
    source_span: str = ""


# ---- Client protocols (injected; fakes in tests) ----

class Embedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class ChatLLM(Protocol):
    def complete(self, prompt: str) -> str: ...
    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel: ...


class VisionLLM(Protocol):
    def ocr_image(self, data: bytes, mime: str) -> str: ...


@dataclass
class Deps:
    chat: ChatLLM
    vision: VisionLLM
    embedder: Embedder
    session_factory: Any  # callable() -> SQLAlchemy Session


# ---- Graph state ----

class AgentState(TypedDict, total=False):
    messages: list[dict[str, Any]]
    intent: str | None
    # ingest
    document_id: int | None
    file_path: str | None
    ocr_text: str | None
    extracted: dict[str, Any] | None      # ExtractionResult.model_dump()
    patient_id: int | None
    patient_candidates: list[dict[str, Any]]
    # query
    query_filters: dict[str, Any] | None
    retrieved: list[dict[str, Any]]
    answer: str | None
    citations: list[dict[str, Any]]
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_state.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/__init__.py app/agent/state.py tests/agent/__init__.py tests/agent/test_state.py
git commit -m "feat: agent state, extraction schemas, client protocols, Deps"
```

---

## Task 3: Embeddings client (Ollama)

**Files:**
- Create: `app/agent/embeddings.py`
- Test: `tests/agent/test_embeddings.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_embeddings.py`:

```python
from app.agent.embeddings import OllamaEmbedder


class _FakeInner:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_embedder_delegates_and_dims():
    emb = OllamaEmbedder(inner=_FakeInner())
    assert len(emb.embed_query("hi")) == 768
    assert len(emb.embed_documents(["a", "b"])) == 2
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/embeddings.py`:

```python
from __future__ import annotations

from app.config import get_settings


class OllamaEmbedder:
    """Wraps langchain_ollama.OllamaEmbeddings; `inner` injectable for tests."""

    def __init__(self, inner=None):
        if inner is None:
            from langchain_ollama import OllamaEmbeddings
            s = get_settings()
            inner = OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_host)
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_embeddings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/embeddings.py tests/agent/test_embeddings.py
git commit -m "feat: Ollama embedder wrapper (nomic-embed-text, 768-dim)"
```

---

## Task 4: LLM clients (Groq chat + vision)

**Files:**
- Create: `app/agent/llm.py`
- Test: `tests/agent/test_llm.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_llm.py`:

```python
from pydantic import BaseModel
from app.agent.llm import GroqChat


class _Schema(BaseModel):
    answer: str


class _FakeLC:
    def invoke(self, prompt):
        class _R:
            content = "hello world"
        return _R()

    def with_structured_output(self, schema):
        class _S:
            def invoke(self, prompt):
                return schema(answer="grounded")
        return _S()


def test_chat_complete_returns_text():
    chat = GroqChat(inner=_FakeLC())
    assert chat.complete("hi") == "hello world"


def test_chat_structured_returns_model():
    chat = GroqChat(inner=_FakeLC())
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema)
    assert out.answer == "grounded"
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/llm.py`:

```python
from __future__ import annotations

import base64

from pydantic import BaseModel

from app.config import get_settings


class GroqChat:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            inner = ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0)
        self._inner = inner

    def complete(self, prompt: str) -> str:
        return self._inner.invoke(prompt).content

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self._inner.with_structured_output(schema).invoke(prompt)


class GroqVision:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            inner = ChatGroq(model=s.groq_vision_model, api_key=s.groq_api_key, temperature=0)
        self._inner = inner

    def ocr_image(self, data: bytes, mime: str) -> str:
        b64 = base64.b64encode(data).decode()
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe ALL text in this medical document verbatim. Output only the text."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        return self._inner.invoke(msg).content
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/llm.py tests/agent/test_llm.py
git commit -m "feat: Groq chat + vision client wrappers"
```

---

## Task 5: Extraction service (pypdf + vision OCR)

**Files:**
- Create: `app/services/extraction.py`
- Test: `tests/test_extraction_service.py`

- [ ] **Step 1: Write the failing test**

`tests/test_extraction_service.py`:

```python
from app.services.extraction import extract_text


class _FakeVision:
    def ocr_image(self, data, mime):
        return "OCR: Patient Jane Doe"


def test_image_routes_to_vision():
    text = extract_text(b"\x89PNG fake", mime_type="image/png", vision=_FakeVision())
    assert text == "OCR: Patient Jane Doe"


def test_plain_text_passthrough():
    text = extract_text(b"hello report", mime_type="text/plain", vision=_FakeVision())
    assert text == "hello report"


def test_text_pdf_uses_pypdf(tmp_path):
    # minimal one-page text PDF built with pypdf
    from pypdf import PdfWriter
    import io
    w = PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    # blank PDF -> empty/whitespace extracted text -> falls back to vision
    out = extract_text(buf.getvalue(), mime_type="application/pdf", vision=_FakeVision())
    assert out == "OCR: Patient Jane Doe"
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_extraction_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/services/extraction.py`:

```python
from __future__ import annotations

import io


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = [(page.extract_text() or "") for page in reader.pages]
    return "\n".join(parts).strip()


def extract_text(data: bytes, *, mime_type: str, vision) -> str:
    """Return document text. Text PDFs via pypdf; images and scanned PDFs via Groq vision.

    `vision` is a VisionLLM (injected).
    """
    if mime_type == "text/plain":
        return data.decode("utf-8", errors="replace")
    if mime_type == "application/pdf":
        text = _pdf_text(data)
        if text:
            return text
        # scanned PDF (no text layer) -> OCR first page bytes as image fallback
        return vision.ocr_image(data, "application/pdf")
    if mime_type.startswith("image/"):
        return vision.ocr_image(data, mime_type)
    raise ValueError(f"unsupported mime_type: {mime_type}")
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_extraction_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/extraction.py tests/test_extraction_service.py
git commit -m "feat: extraction service (pypdf text + Groq vision OCR fallback)"
```

---

## Task 6: Entities service (persist + links)

**Files:**
- Create: `app/services/entities.py`
- Test: `tests/test_entities_service.py`

DB-touching test — uses the existing `db` fixture from `tests/conftest.py` (Supabase pooler session). Inspect `tests/conftest.py` for the fixture name before writing; this plan assumes a `db` Session fixture (as `tests/test_documents_service.py` uses).

- [ ] **Step 1: Write the failing test**

`tests/test_entities_service.py`:

```python
from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.entities import persist_extraction
from app.agent.state import ExtractionResult, ExtractedEntity, ExtractedTest
from app.models import Disease, DocumentEntity


def test_persist_creates_entities_and_links(db):
    p = create_patient(db, name="Persist Test")
    doc = create_document(db, patient_id=p.id, doc_type="prescription")
    er = ExtractionResult(
        patient_name="Persist Test", doc_type="prescription",
        diseases=[ExtractedEntity(name="asthma", confidence=0.9, source_span="Dx asthma")],
        medications=[ExtractedEntity(name="salbutamol", confidence=0.8, source_span="Rx salbutamol")],
        tests=[ExtractedTest(name="spirometry", value="80", unit="%")],
    )
    n = persist_extraction(db, document_id=doc.id, result=er)
    assert n >= 3  # disease + medication + test linked
    links = db.query(DocumentEntity).filter_by(document_id=doc.id).all()
    assert any(l.entity_type == "disease" and l.validated for l in links)
    # entity de-duplicated by name
    assert db.query(Disease).filter(Disease.name == "asthma").count() == 1
    persist_extraction(db, document_id=doc.id, result=er)
    assert db.query(Disease).filter(Disease.name == "asthma").count() == 1
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_entities_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/services/entities.py`:

```python
from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.agent.state import ExtractionResult
from app.models import (
    Disease, DocumentEntity, Doctor, Medication, MedicalTest, Symptom, TestResult,
)


def _upsert_by_name(db: Session, model, name: str):
    obj = db.query(model).filter(func.lower(model.name) == name.lower()).first()
    if obj is None:
        obj = model(name=name)
        db.add(obj)
        db.flush()
    return obj


def _link(db: Session, document_id: int, entity_type: str, entity_id: int,
          confidence: float, source_span: str) -> None:
    db.add(DocumentEntity(
        document_id=document_id, entity_type=entity_type, entity_id=entity_id,
        confidence=confidence, source_span=source_span, validated=True,
    ))


def persist_extraction(db: Session, *, document_id: int, result: ExtractionResult) -> int:
    """Upsert extracted entities (by name) and link them to the document. Returns link count."""
    count = 0
    if result.doctor:
        d = _upsert_by_name(db, Doctor, result.doctor)
        _link(db, document_id, "doctor", d.id, result.confidence, result.source_span)
        count += 1
    for e in result.diseases:
        obj = _upsert_by_name(db, Disease, e.name)
        _link(db, document_id, "disease", obj.id, e.confidence, e.source_span)
        count += 1
    for e in result.symptoms:
        obj = _upsert_by_name(db, Symptom, e.name)
        _link(db, document_id, "symptom", obj.id, e.confidence, e.source_span)
        count += 1
    for e in result.medications:
        obj = _upsert_by_name(db, Medication, e.name)
        _link(db, document_id, "medication", obj.id, e.confidence, e.source_span)
        count += 1
    for t in result.tests:
        mt = _upsert_by_name(db, MedicalTest, t.name)
        tr = TestResult(medical_test_id=mt.id, value=t.value, unit=t.unit,
                        reference_range=t.reference_range)
        db.add(tr)
        db.flush()
        _link(db, document_id, "test_result", tr.id, t.confidence, t.source_span)
        count += 1
    db.commit()
    return count
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_entities_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/entities.py tests/test_entities_service.py
git commit -m "feat: entities service (name-dedup upsert + validated document links)"
```

---

## Task 7: Chunking service (chunk + embed + persist)

**Files:**
- Create: `app/services/chunking.py`
- Test: `tests/test_chunking_service.py`

- [ ] **Step 1: Write the failing test**

`tests/test_chunking_service.py`:

```python
from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import make_chunks, chunk_and_embed
from app.models import Chunk


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.2] * 768 for _ in texts]


def test_make_chunks_prefixes_header():
    chunks = make_chunks("line one. line two.", header="Jane · prescription · 2026-06-10", size=12, overlap=0)
    assert len(chunks) >= 2
    assert all(c.startswith("[Jane · prescription · 2026-06-10]") for c in chunks)


def test_chunk_and_embed_persists(db):
    p = create_patient(db, name="Chunk Test")
    doc = create_document(db, patient_id=p.id, doc_type="lab_report")
    n = chunk_and_embed(
        db, document_id=doc.id, patient_id=p.id,
        text="hemoglobin 13.5 normal. wbc 6000 normal.",
        header="Chunk Test · lab_report · 2026-06-10",
        embedder=_FakeEmbedder(), size=20, overlap=0,
    )
    rows = db.query(Chunk).filter_by(document_id=doc.id).all()
    assert n == len(rows) >= 2
    assert rows[0].patient_id == p.id
    assert len(rows[0].embedding) == 768
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_chunking_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/services/chunking.py`:

```python
from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Chunk


def make_chunks(text: str, *, header: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Sliding-window character chunks, each prefixed with a contextual header."""
    text = " ".join(text.split())
    if not text:
        return []
    out: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        body = text[start:start + size]
        out.append(f"[{header}] {body}")
        start += step
    return out


def chunk_and_embed(db: Session, *, document_id: int, patient_id: int, text: str,
                    header: str, embedder, size: int = 800, overlap: int = 100) -> int:
    """Chunk text, embed each chunk, persist Chunk rows (with denormalized patient_id). Returns count."""
    chunks = make_chunks(text, header=header, size=size, overlap=overlap)
    if not chunks:
        return 0
    vectors = embedder.embed_documents(chunks)
    for i, (body, vec) in enumerate(zip(chunks, vectors)):
        db.add(Chunk(document_id=document_id, patient_id=patient_id, ord=i,
                     text=body, embedding=vec))
    db.commit()
    return len(chunks)
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_chunking_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/chunking.py tests/test_chunking_service.py
git commit -m "feat: chunking service (header chunks + embed + persist with patient_id)"
```

---

## Task 8: Retrieval service (pgvector, patient-scoped)

**Files:**
- Create: `app/services/retrieval.py`
- Test: `tests/test_retrieval_service.py`

- [ ] **Step 1: Write the failing test**

`tests/test_retrieval_service.py`:

```python
from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import chunk_and_embed
from app.services.retrieval import search_chunks


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.2] * 768

    def embed_documents(self, texts):
        return [[0.2] * 768 for _ in texts]


def test_search_is_patient_scoped(db):
    pa = create_patient(db, name="Alice Scope")
    pb = create_patient(db, name="Bob Scope")
    da = create_document(db, patient_id=pa.id, doc_type="lab_report")
    dbb = create_document(db, patient_id=pb.id, doc_type="lab_report")
    emb = _FakeEmbedder()
    chunk_and_embed(db, document_id=da.id, patient_id=pa.id,
                    text="alice hemoglobin 13", header="Alice", embedder=emb, size=50)
    chunk_and_embed(db, document_id=dbb.id, patient_id=pb.id,
                    text="bob hemoglobin 14", header="Bob", embedder=emb, size=50)

    hits = search_chunks(db, patient_id=pa.id, query="hemoglobin", embedder=emb, k=5)
    assert hits, "expected at least one hit"
    assert all(h["patient_id"] == pa.id for h in hits)
    assert all("chunk_id" in h and "text" in h for h in hits)
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/test_retrieval_service.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/services/retrieval.py`:

```python
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document


def search_chunks(db: Session, *, patient_id: int, query: str, embedder, k: int = 5) -> list[dict]:
    """Patient-scoped pgvector cosine search. Returns dicts with proof metadata."""
    qvec = embedder.embed_query(query)
    stmt = (
        select(Chunk, Document.doc_type, Document.uploaded_at)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.patient_id == patient_id)
        .order_by(Chunk.embedding.cosine_distance(qvec))
        .limit(k)
    )
    rows = db.execute(stmt).all()
    out: list[dict] = []
    for chunk, doc_type, uploaded_at in rows:
        out.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "patient_id": chunk.patient_id,
            "text": chunk.text,
            "doc_type": doc_type,
            "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
        })
    return out
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/test_retrieval_service.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/services/retrieval.py tests/test_retrieval_service.py
git commit -m "feat: retrieval service (pgvector cosine, patient-scoped, proof metadata)"
```

---

## Task 9: Router node

**Files:**
- Create: `app/agent/router.py`
- Test: `tests/agent/test_router.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_router.py`:

```python
from app.agent.router import classify_intent


class _FakeChat:
    def __init__(self, label):
        self._label = label

    def complete(self, prompt):
        return self._label

    def structured(self, prompt, schema):
        raise NotImplementedError


def _cfg(chat):
    from app.agent.state import Deps
    deps = Deps(chat=chat, vision=None, embedder=None, session_factory=None)
    return {"configurable": {"deps": deps}}


def test_router_ingest_when_file_present():
    state = {"messages": [{"role": "user", "content": "read this"}], "file_path": "/x.png"}
    out = classify_intent(state, _cfg(_FakeChat("rag_query")))
    assert out["intent"] == "ingest"  # file present forces ingest


def test_router_uses_llm_label_for_text():
    state = {"messages": [{"role": "user", "content": "latest report of Jane"}]}
    out = classify_intent(state, _cfg(_FakeChat("structured_query")))
    assert out["intent"] == "structured_query"


def test_router_defaults_to_rag_on_garbage():
    state = {"messages": [{"role": "user", "content": "what about her sugar?"}]}
    out = classify_intent(state, _cfg(_FakeChat("nonsense")))
    assert out["intent"] == "rag_query"
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_router.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/router.py`:

```python
from __future__ import annotations

from typing import Any

_VALID = {"ingest", "structured_query", "rag_query"}

_PROMPT = """Classify the user's request into exactly one label:
- structured_query: asking for a specific document/record by patient, type, or recency (e.g. "latest report of Jane", "show prescriptions for Bob").
- rag_query: a question about the CONTENT of documents (e.g. "what did the doctor say about her blood pressure?").
Respond with ONLY the label.

User: {text}
Label:"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def classify_intent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    # A pending file upload always means ingest.
    if state.get("file_path"):
        return {"intent": "ingest"}
    deps = config["configurable"]["deps"]
    label = deps.chat.complete(_PROMPT.format(text=_last_user_text(state))).strip().lower()
    if "structured" in label:
        return {"intent": "structured_query"}
    if "ingest" in label:
        return {"intent": "ingest"}
    return {"intent": "rag_query"}  # safe default
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_router.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/router.py tests/agent/test_router.py
git commit -m "feat: router node (file->ingest, LLM intent classify, rag default)"
```

---

## Task 10: Ingest nodes

**Files:**
- Create: `app/agent/nodes/__init__.py` (empty)
- Create: `app/agent/nodes/ingest.py`
- Test: `tests/agent/test_ingest_nodes.py`

Nodes are split so each is independently testable. The HITL `interrupt()` lives only in `confirm_entities_node` and `confirm_patient_node`; pure-logic nodes are tested directly.

- [ ] **Step 1: Write the failing test**

`tests/agent/test_ingest_nodes.py`:

```python
from app.agent.state import Deps, ExtractionResult, ExtractedEntity
from app.agent.nodes.ingest import (
    extract_text_node, extract_entities_node, resolve_patient_node,
)


class _FakeVision:
    def ocr_image(self, data, mime):
        return "Patient Jane Doe, Dx hypertension"


class _FakeChat:
    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
        return ExtractionResult(
            patient_name="Jane Doe", doc_type="prescription",
            diseases=[ExtractedEntity(name="hypertension", confidence=0.9, source_span="Dx hypertension")],
        )


def _cfg(**kw):
    deps = Deps(chat=kw.get("chat"), vision=kw.get("vision"),
                embedder=kw.get("embedder"), session_factory=kw.get("sf"))
    return {"configurable": {"deps": deps}}


def test_extract_text_node_reads_bytes(tmp_path):
    f = tmp_path / "scan.png"
    f.write_bytes(b"\x89PNG fake")
    state = {"file_path": str(f), "mime_type": "image/png"}
    out = extract_text_node(state, _cfg(vision=_FakeVision()))
    assert "Jane Doe" in out["ocr_text"]


def test_extract_entities_node_returns_dict():
    state = {"ocr_text": "Patient Jane Doe, Dx hypertension"}
    out = extract_entities_node(state, _cfg(chat=_FakeChat()))
    assert out["extracted"]["patient_name"] == "Jane Doe"
    assert out["extracted"]["diseases"][0]["name"] == "hypertension"


def test_resolve_patient_exact_match(db_session_factory):
    from app.services.patients import create_patient
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Unique Resolve Name")
    state = {"extracted": {"patient_name": "Unique Resolve Name"}}
    out = resolve_patient_node(state, _cfg(sf=sf))
    assert out["patient_id"] == p.id
    assert out.get("patient_candidates") in (None, [])


def test_resolve_patient_no_match_sets_candidates(db_session_factory):
    state = {"extracted": {"patient_name": "Nobody Named This 9999"}}
    out = resolve_patient_node(state, _cfg(sf=db_session_factory))
    assert out["patient_id"] is None
    assert out["patient_candidates"] == []
```

Note: add a `db_session_factory` fixture to `tests/conftest.py` if absent — a callable returning a new Session context manager bound to the test engine. Also `extract_text_node` expects `state["mime_type"]`; the Streamlit page (Task 13) sets it on upload.

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_ingest_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/nodes/ingest.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from app.agent.state import ExtractionResult
from app.services.chunking import chunk_and_embed
from app.services.documents import get_document
from app.services.entities import persist_extraction
from app.services.extraction import extract_text
from app.models import Patient


_EXTRACT_PROMPT = """Extract structured medical data from this document text.
For each entity set confidence (0-1) and source_span (the exact text you used).
If a field is absent, leave it null/empty.

Document:
{text}"""


def extract_text_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    data = Path(state["file_path"]).read_bytes()
    text = extract_text(data, mime_type=state["mime_type"], vision=deps.vision)
    return {"ocr_text": text}


def extract_entities_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    result = deps.chat.structured(
        _EXTRACT_PROMPT.format(text=state["ocr_text"]), ExtractionResult
    )
    return {"extracted": result.model_dump()}


def confirm_entities_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate: human reviews/edits extracted entities; same approval commits the write."""
    decision = interrupt({"type": "confirm_entities", "extracted": state["extracted"]})
    if not decision.get("approved"):
        return {"extracted": None, "intent": "rejected"}
    # human may return edited entities
    return {"extracted": decision.get("extracted", state["extracted"])}


def resolve_patient_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    name = (state.get("extracted") or {}).get("patient_name")
    if not name:
        return {"patient_id": None, "patient_candidates": []}
    with deps.session_factory() as s:
        matches = (
            s.query(Patient)
            .filter(Patient.name.ilike(name))
            .all()
        )
        cands = [{"id": p.id, "name": p.name} for p in matches]
    if len(cands) == 1:
        return {"patient_id": cands[0]["id"], "patient_candidates": []}
    return {"patient_id": None, "patient_candidates": cands}


def confirm_patient_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate (only reached when patient ambiguous): pick existing id or create new."""
    if state.get("patient_id"):
        return {}
    deps = config["configurable"]["deps"]
    decision = interrupt({
        "type": "confirm_patient",
        "candidates": state.get("patient_candidates", []),
        "extracted_name": (state.get("extracted") or {}).get("patient_name"),
    })
    pid = decision.get("patient_id")
    if pid is None and decision.get("create_new"):
        from app.services.patients import create_patient
        with deps.session_factory() as s:
            p = create_patient(s, name=(state.get("extracted") or {}).get("patient_name") or "Unknown")
            pid = p.id
    return {"patient_id": pid}


def persist_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    result = ExtractionResult(**state["extracted"])
    with deps.session_factory() as s:
        n = persist_extraction(s, document_id=state["document_id"], result=result)
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Saved {n} entities for this document."}
    ]}


def chunk_embed_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    ex = state["extracted"]
    header = f"{ex.get('patient_name') or ''} · {ex.get('doc_type') or 'doc'} · {ex.get('doc_date') or ''}".strip()
    with deps.session_factory() as s:
        doc = get_document(s, state["document_id"])
        if doc is not None and state.get("ocr_text"):
            doc.raw_ocr_text = state["ocr_text"]
            doc.status = "indexed"
            s.commit()
        n = chunk_and_embed(
            s, document_id=state["document_id"], patient_id=state["patient_id"],
            text=state.get("ocr_text") or "", header=header, embedder=deps.embedder,
        )
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Indexed {n} chunks. Ingestion complete."}
    ]}
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_ingest_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes/__init__.py app/agent/nodes/ingest.py tests/agent/test_ingest_nodes.py
git commit -m "feat: ingest nodes (extract text/entities, resolve patient, persist, chunk+embed, HITL gates)"
```

---

## Task 11: Structured-query nodes

**Files:**
- Create: `app/agent/nodes/structured.py`
- Test: `tests/agent/test_structured_nodes.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_structured_nodes.py`:

```python
from pydantic import BaseModel
from app.agent.state import Deps
from app.agent.nodes.structured import parse_filters_node, query_db_node


class _FakeChat:
    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
        return schema(**self._payload)


def _cfg(chat=None, sf=None):
    return {"configurable": {"deps": Deps(chat=chat, vision=None, embedder=None, session_factory=sf)}}


def test_parse_filters_extracts_name_and_recency():
    state = {"messages": [{"role": "user", "content": "latest report of Jane Doe"}]}
    chat = _FakeChat({"patient_name": "Jane Doe", "doc_type": None, "latest": True})
    out = parse_filters_node(state, _cfg(chat=chat))
    assert out["query_filters"]["patient_name"] == "Jane Doe"
    assert out["query_filters"]["latest"] is True


def test_query_db_returns_latest_document(db_session_factory):
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Latest Query Pt")
        create_document(s, patient_id=p.id, doc_type="lab_report")
        create_document(s, patient_id=p.id, doc_type="prescription")
    state = {"query_filters": {"patient_name": "Latest Query Pt", "doc_type": None, "latest": True}}
    out = query_db_node(state, _cfg(sf=sf))
    assert "prescription" in out["answer"] or "lab_report" in out["answer"]
    assert out["citations"]
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_structured_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/nodes/structured.py`:

```python
from __future__ import annotations

from typing import Any

from pydantic import BaseModel
from sqlalchemy import func

from app.agent.state import Deps  # noqa: F401  (documents the dep contract)
from app.models import Document, Patient


class _Filters(BaseModel):
    patient_name: str | None = None
    doc_type: str | None = None
    latest: bool = False


_PROMPT = """From the user's request extract: patient_name, doc_type (or null), and
latest (true if they want the most recent/last one). Request: {text}"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def parse_filters_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = deps.chat.structured(_PROMPT.format(text=_last_user_text(state)), _Filters)
    return {"query_filters": f.model_dump()}


def query_db_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    f = state["query_filters"]
    with deps.session_factory() as s:
        q = s.query(Document).join(Patient, Patient.id == Document.patient_id)
        if f.get("patient_name"):
            q = q.filter(func.lower(Patient.name) == f["patient_name"].lower())
        if f.get("doc_type"):
            q = q.filter(Document.doc_type == f["doc_type"])
        q = q.order_by(Document.uploaded_at.desc(), Document.id.desc())
        limit = 1 if f.get("latest") else 10
        docs = q.limit(limit).all()
        rows = [{"document_id": d.id, "doc_type": d.doc_type,
                 "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None}
                for d in docs]
    if not rows:
        return {"answer": "No matching documents found.", "citations": []}
    lines = [f"- {r['doc_type'] or 'document'} (id {r['document_id']}, {r['uploaded_at']})" for r in rows]
    return {"answer": "Found:\n" + "\n".join(lines), "citations": rows}
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_structured_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes/structured.py tests/agent/test_structured_nodes.py
git commit -m "feat: structured-query nodes (LLM filter parse + latest/list SQL)"
```

---

## Task 12: RAG nodes

**Files:**
- Create: `app/agent/nodes/rag.py`
- Test: `tests/agent/test_rag_nodes.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_rag_nodes.py`:

```python
from app.agent.state import Deps
from app.agent.nodes.rag import grade_node, generate_answer_node


class _FakeChat:
    def __init__(self, text):
        self._text = text

    def complete(self, prompt):
        return self._text

    def structured(self, prompt, schema):
        raise NotImplementedError


def _cfg(chat):
    return {"configurable": {"deps": Deps(chat=chat, vision=None, embedder=None, session_factory=None)}}


def test_grade_low_confidence_flags():
    state = {"retrieved": [{"chunk_id": 1, "text": "x"}]}
    out = grade_node(state, _cfg(_FakeChat("0.1")))
    assert out["low_confidence"] is True


def test_grade_high_confidence_passes():
    state = {"retrieved": [{"chunk_id": 1, "text": "x"}]}
    out = grade_node(state, _cfg(_FakeChat("0.92")))
    assert out["low_confidence"] is False


def test_generate_answer_includes_citations():
    state = {
        "messages": [{"role": "user", "content": "what is her hemoglobin?"}],
        "retrieved": [
            {"chunk_id": 7, "text": "hemoglobin 13.5 g/dL", "doc_type": "lab_report", "uploaded_at": "2026-06-10"},
        ],
    }
    out = generate_answer_node(state, _cfg(_FakeChat("Her hemoglobin is 13.5 g/dL")))
    assert out["answer"]
    assert out["citations"][0]["chunk_id"] == 7
    assert "#7" in out["messages"][-1]["content"]


def test_generate_answer_refuses_when_empty():
    state = {"messages": [{"role": "user", "content": "x"}], "retrieved": []}
    out = generate_answer_node(state, _cfg(_FakeChat("ignored")))
    assert "don't have" in out["answer"].lower() or "no relevant" in out["answer"].lower()
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_rag_nodes.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/nodes/rag.py`:

```python
from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.config import get_settings
from app.services.retrieval import search_chunks

_GRADE_PROMPT = """Rate 0.0-1.0 how well these snippets can answer the question.
Respond with ONLY a number.
Question: {q}
Snippets:
{snips}"""

_ANSWER_PROMPT = """Answer the question USING ONLY the snippets. Do not use outside knowledge.
If the snippets don't contain the answer, say you don't have that information.
Question: {q}
Snippets:
{snips}"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def retrieve_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    s = get_settings()
    with deps.session_factory() as sess:
        hits = search_chunks(sess, patient_id=state["patient_id"],
                             query=_last_user_text(state), embedder=deps.embedder,
                             k=s.rag_top_k)
    return {"retrieved": hits}


def grade_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        return {"low_confidence": True}
    snips = "\n".join(f"[#{h['chunk_id']}] {h['text']}" for h in hits)
    raw = deps.chat.complete(_GRADE_PROMPT.format(q=_last_user_text(state), snips=snips))
    try:
        score = float(raw.strip().split()[0])
    except (ValueError, IndexError):
        score = 0.0
    threshold = get_settings().rag_confidence_threshold
    return {"low_confidence": score < threshold, "grade_score": score}


def confirm_low_confidence_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate: weak retrieval -> ask whether to answer anyway."""
    if not state.get("low_confidence"):
        return {}
    decision = interrupt({
        "type": "low_confidence",
        "score": state.get("grade_score"),
        "snippets": state.get("retrieved", []),
    })
    if not decision.get("proceed"):
        return {"retrieved": []}  # forces a refusal downstream
    return {}


def generate_answer_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        msg = "I don't have relevant information in this patient's documents to answer that."
        return {"answer": msg,
                "citations": [],
                "messages": state["messages"] + [{"role": "assistant", "content": msg}]}
    snips = "\n".join(f"[#{h['chunk_id']}] {h['text']}" for h in hits)
    body = deps.chat.complete(_ANSWER_PROMPT.format(q=_last_user_text(state), snips=snips))
    cites = "\n".join(
        f"  - #{h['chunk_id']} ({h.get('doc_type') or 'doc'}, {h.get('uploaded_at') or ''}): "
        f"\"{h['text'][:120]}\"" for h in hits
    )
    full = f"{body}\n\nSources:\n{cites}"
    return {"answer": body, "citations": hits,
            "messages": state["messages"] + [{"role": "assistant", "content": full}]}
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_rag_nodes.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/agent/nodes/rag.py tests/agent/test_rag_nodes.py
git commit -m "feat: RAG nodes (retrieve, grade, low-confidence HITL, grounded answer w/ citations)"
```

---

## Task 13: Graph wiring + interrupt/resume integration test

**Files:**
- Create: `app/agent/graph.py`
- Test: `tests/agent/test_graph.py`

- [ ] **Step 1: Write the failing test**

`tests/agent/test_graph.py`:

```python
import uuid
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.state import Deps, ExtractionResult, ExtractedEntity


class _FakeChat:
    def __init__(self, label):
        self._label = label

    def complete(self, prompt):
        return self._label

    def structured(self, prompt, schema):
        if schema is ExtractionResult:
            return ExtractionResult(
                patient_name="Graph Pt", doc_type="prescription",
                diseases=[ExtractedEntity(name="flu", confidence=0.9, source_span="flu")],
            )
        return schema()


class _FakeVision:
    def ocr_image(self, data, mime):
        return "Patient Graph Pt, flu"


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_ingest_pauses_at_confirm_entities(db_session_factory, tmp_path):
    from app.services.patients import create_patient
    from app.services.documents import create_document
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Graph Pt")
        doc = create_document(s, patient_id=p.id, doc_type="prescription")
    f = tmp_path / "rx.png"
    f.write_bytes(b"\x89PNG")
    deps = Deps(chat=_FakeChat("ingest"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    state = {"messages": [{"role": "user", "content": "read this"}],
             "file_path": str(f), "mime_type": "image/png", "document_id": doc.id}

    result = graph.invoke(state, cfg)
    assert "__interrupt__" in result  # paused at confirm_entities

    # human approves -> resume runs to completion (resolves Graph Pt exactly, persists, indexes)
    final = graph.invoke(Command(resume={"approved": True}), cfg)
    assert final["patient_id"] == p.id
    assert any("Indexed" in m["content"] for m in final["messages"])


def test_rag_query_runs_without_pause(db_session_factory):
    from app.services.patients import create_patient
    sf = db_session_factory
    with sf() as s:
        p = create_patient(s, name="Rag Pt")
    deps = Deps(chat=_FakeChat("0.9"), vision=_FakeVision(),
                embedder=_FakeEmbedder(), session_factory=sf)
    graph = build_graph()
    cfg = {"configurable": {"deps": deps, "thread_id": str(uuid.uuid4())}}
    state = {"messages": [{"role": "user", "content": "what about flu?"}],
             "patient_id": p.id}
    out = graph.invoke(state, cfg)
    # no chunks for this patient -> graceful refusal, no crash
    assert out["answer"]
```

- [ ] **Step 2: Run, verify fail**

Run: `pytest tests/agent/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`app/agent/graph.py`:

```python
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.router import classify_intent
from app.agent.state import AgentState
from app.agent.nodes.ingest import (
    chunk_embed_node, confirm_entities_node, confirm_patient_node,
    extract_entities_node, extract_text_node, persist_node, resolve_patient_node,
)
from app.agent.nodes.structured import parse_filters_node, query_db_node
from app.agent.nodes.rag import (
    confirm_low_confidence_node, generate_answer_node, grade_node, retrieve_node,
)


def _route(state: AgentState) -> str:
    return state.get("intent") or "rag_query"


def _after_confirm_entities(state: AgentState) -> str:
    # rejected -> stop; else continue to patient resolution
    return "rejected" if state.get("intent") == "rejected" else "resolve_patient"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("router", classify_intent)
    # ingest
    g.add_node("extract_text", extract_text_node)
    g.add_node("extract_entities", extract_entities_node)
    g.add_node("confirm_entities", confirm_entities_node)
    g.add_node("resolve_patient", resolve_patient_node)
    g.add_node("confirm_patient", confirm_patient_node)
    g.add_node("persist", persist_node)
    g.add_node("chunk_embed", chunk_embed_node)
    # structured
    g.add_node("parse_filters", parse_filters_node)
    g.add_node("query_db", query_db_node)
    # rag
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("confirm_low_confidence", confirm_low_confidence_node)
    g.add_node("generate_answer", generate_answer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route, {
        "ingest": "extract_text",
        "structured_query": "parse_filters",
        "rag_query": "retrieve",
    })

    # ingest chain
    g.add_edge("extract_text", "extract_entities")
    g.add_edge("extract_entities", "confirm_entities")
    g.add_conditional_edges("confirm_entities", _after_confirm_entities, {
        "rejected": END, "resolve_patient": "resolve_patient",
    })
    g.add_edge("resolve_patient", "confirm_patient")
    g.add_edge("confirm_patient", "persist")
    g.add_edge("persist", "chunk_embed")
    g.add_edge("chunk_embed", END)

    # structured chain
    g.add_edge("parse_filters", "query_db")
    g.add_edge("query_db", END)

    # rag chain
    g.add_edge("retrieve", "grade")
    g.add_edge("grade", "confirm_low_confidence")
    g.add_edge("confirm_low_confidence", "generate_answer")
    g.add_edge("generate_answer", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
```

- [ ] **Step 4: Run, verify pass**

Run: `pytest tests/agent/test_graph.py -v`
Expected: PASS

(If `confirm_patient` interrupts for the exact-match case, it must early-return `{}` when `patient_id` is set — verify Task 10's `confirm_patient_node` guard. The exact-match test patient "Graph Pt" resolves to one row, so no interrupt fires there.)

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/agent/test_graph.py
git commit -m "feat: build_graph (supervisor wiring, MemorySaver, interrupt/resume)"
```

---

## Task 14: Streamlit Chat page

**Files:**
- Modify: `streamlit_app.py`
- Manual verification (no unit test — UI glue).

- [ ] **Step 1: Add a deps builder + Chat page**

Add to `streamlit_app.py` (integrate with existing page nav; if the app is single-page, add a `st.sidebar.radio` page switch or a new tab):

```python
import uuid
import streamlit as st

from app.agent.graph import build_graph
from app.agent.llm import GroqChat, GroqVision
from app.agent.embeddings import OllamaEmbedder
from app.agent.state import Deps
from app.db import SessionLocal  # existing session factory in app/db.py
from app.services.documents import create_document
from app.storage import save_bytes
from langgraph.types import Command


@st.cache_resource
def _graph():
    return build_graph()


def _deps() -> Deps:
    return Deps(chat=GroqChat(), vision=GroqVision(),
                embedder=OllamaEmbedder(), session_factory=SessionLocal)


def chat_page():
    st.header("Agent Chat")
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    cfg = {"configurable": {"deps": _deps(), "thread_id": st.session_state.thread_id}}
    graph = _graph()

    # Render an active interrupt (HITL gate), if any.
    pending = st.session_state.get("pending_interrupt")
    if pending:
        _render_interrupt(graph, cfg, pending)
        return

    for m in st.session_state.chat_log:
        st.chat_message(m["role"]).write(m["content"])

    up = st.file_uploader("Attach a medical document (optional)", type=["png", "jpg", "jpeg", "pdf", "txt"])
    prompt = st.chat_input("Ask, or upload + 'read this and arrange it'")
    if not prompt:
        return

    state = {"messages": st.session_state.chat_log + [{"role": "user", "content": prompt}]}
    if up is not None:
        data = up.getvalue()
        with SessionLocal() as s:
            # patient_id 0 placeholder; resolve_patient sets the real one. Use existing default patient flow if needed.
            doc = create_document(s, patient_id=_default_patient_id(), mime_type=up.type)
            path = save_bytes(doc.patient_id, doc.id, up.name.rsplit(".", 1)[-1], data)
            doc_id = doc.id
        state.update({"file_path": path, "mime_type": up.type, "document_id": doc_id})

    _run(graph, cfg, state)


def _run(graph, cfg, payload):
    result = graph.invoke(payload, cfg)
    _absorb(graph, cfg, result)


def _absorb(graph, cfg, result):
    if "__interrupt__" in result:
        intr = result["__interrupt__"][0]
        st.session_state.pending_interrupt = intr.value
        st.rerun()
    st.session_state.chat_log = result.get("messages", st.session_state.chat_log)
    st.rerun()


def _render_interrupt(graph, cfg, payload):
    kind = payload.get("type")
    st.warning(f"Action needs your approval: {kind}")
    if kind == "confirm_entities":
        st.json(payload["extracted"])
        c1, c2 = st.columns(2)
        if c1.button("Approve & save"):
            st.session_state.pending_interrupt = None
            _run_resume(graph, cfg, {"approved": True, "extracted": payload["extracted"]})
        if c2.button("Reject"):
            st.session_state.pending_interrupt = None
            _run_resume(graph, cfg, {"approved": False})
    elif kind == "confirm_patient":
        st.write("Which patient?", payload.get("candidates"))
        choice = st.text_input("Existing patient id (blank = create new)")
        if st.button("Confirm patient"):
            st.session_state.pending_interrupt = None
            if choice.strip():
                _run_resume(graph, cfg, {"patient_id": int(choice)})
            else:
                _run_resume(graph, cfg, {"create_new": True})
    elif kind == "low_confidence":
        st.write(f"Weak match (score {payload.get('score')}). Answer anyway?")
        c1, c2 = st.columns(2)
        if c1.button("Answer anyway"):
            st.session_state.pending_interrupt = None
            _run_resume(graph, cfg, {"proceed": True})
        if c2.button("Skip"):
            st.session_state.pending_interrupt = None
            _run_resume(graph, cfg, {"proceed": False})


def _run_resume(graph, cfg, value):
    result = graph.invoke(Command(resume=value), cfg)
    _absorb(graph, cfg, result)


def _default_patient_id() -> int:
    # Reuse the dashboard's currently-selected patient if present; else first patient.
    from app.services.patients import list_patients
    if st.session_state.get("selected_patient_id"):
        return st.session_state.selected_patient_id
    with SessionLocal() as s:
        ps = list_patients(s)
    return ps[0].id if ps else 0
```

Wire `chat_page()` into the existing navigation (add "Chat" to the sidebar radio/menu). Confirm `app/db.py` exports `SessionLocal`; if the session factory has another name, use that and update `_deps()` + `db_session_factory` fixture accordingly.

- [ ] **Step 2: Smoke-test manually**

Run: `streamlit run streamlit_app.py`
Checks:
1. "Chat" page loads.
2. Upload a prescription image + "read this and arrange it" → entities shown in a `confirm_entities` panel.
3. Approve → "Indexed N chunks" message.
4. Ask "latest report of <patient>" → structured list.
5. Ask a content question → grounded answer with a `Sources:` block citing `#<chunk_id>`.

- [ ] **Step 3: Commit**

```bash
git add streamlit_app.py
git commit -m "feat: Streamlit Chat page driving the agent graph with HITL panels"
```

---

## Task 15: conftest fixtures + full suite green

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Ensure fixtures exist**

Inspect `tests/conftest.py`. Ensure two fixtures:
- `db` — a Session bound to the test engine (already used by existing service tests).
- `db_session_factory` — a callable returning a fresh Session **context manager** on the test engine, used by node/graph tests.

If `db_session_factory` is absent, add (adapt engine/session names to the existing file):

```python
import pytest
from app.db import SessionLocal


@pytest.fixture
def db_session_factory():
    # Returns the session factory itself; callers use `with db_session_factory() as s:`
    return SessionLocal
```

If tests must run against a transactional rollback engine rather than `SessionLocal`, return a factory bound to that engine instead, so node tests don't pollute the real DB.

- [ ] **Step 2: Run the whole suite**

Run: `pytest -q`
Expected: all green (existing 18 + new agent/service tests).

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "test: db_session_factory fixture for agent node/graph tests"
```

---

## Self-review notes (resolved)

- **Spec coverage:** router (T9) ✓; ingest OCR/extract/persist/chunk (T5–7,10) ✓; structured "latest report" (T11) ✓; reliable-RAG with citations (T12) ✓; 4 HITL gates — confirm_entities folds the write-approval (T10), confirm_patient (T10), low_confidence (T12) ✓; patient-scoped isolation (T8) ✓; Groq vision OCR (T4–5) ✓; Ollama 768-dim embeddings (T3) ✓; graph + checkpointer + interrupt/resume (T13) ✓; Streamlit surface (T14) ✓.
- **Provider consistency:** `nomic-embed-text`→768 matches `EMBED_DIM`/`Vector(768)`; `groq_model`/`groq_vision_model` from config; no Gemini references.
- **Type consistency:** `ExtractionResult`/`ExtractedEntity`/`ExtractedTest`/`Deps`/`Embedder`/`ChatLLM`/`VisionLLM` defined T2, used identically T3–13. Node signature `(state, config)` and `config["configurable"]["deps"]` uniform across all nodes. `search_chunks` return keys (`chunk_id`,`text`,`doc_type`,`uploaded_at`,`patient_id`) match RAG citation use in T12.
- **Assumptions to verify during execution:** session factory name in `app/db.py` (assumed `SessionLocal`); existing `db` fixture in `tests/conftest.py`; how `streamlit_app.py` currently structures navigation. Each task flags the check inline.
```
