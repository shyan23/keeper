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
    file_ext: str | None
    mime_type: str | None
    source_type: str | None
    ocr_text: str | None
    content_hash: str | None              # SHA-256 of file bytes (dedup)
    original_name: str | None             # uploaded filename, stored on the document
    already_ingested: bool                # True when an identical file was found
    extracted: dict[str, Any] | None      # ExtractionResult.model_dump()
    patient_id: int | None
    patient_candidates: list[dict[str, Any]]
    # query
    query_filters: dict[str, Any] | None
    retrieved: list[dict[str, Any]]
    answer: str | None
    citations: list[dict[str, Any]]
    sources: list[dict[str, Any]]         # per-document citations for the UI (no chunk ids)
    retrieval_query: str | None
    corrected: bool
    low_confidence: bool
    grade_score: float
