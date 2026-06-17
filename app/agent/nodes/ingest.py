from __future__ import annotations

import hashlib
import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from app import storage
from app.agent.state import ExtractionResult
from app.cache import get_or_set, make_key
from app.config import get_settings
from app.services.chunking import chunk_and_embed, make_semantic_chunks
from app.services.documents import (
    create_document, find_by_content_hash, get_document, set_file_path,
)
from app.services.entities import persist_extraction
from app.services.dates import parse_doc_date
from app.services.extraction import extract_text
from app.services.patients import create_patient
from app.models import Patient


_EXTRACT_PROMPT = """Extract structured medical data from this document text.
For each entity set confidence (0-1) and source_span (the exact text you used).
If a field is absent, leave it null/empty.

Document:
{text}"""


def extract_text_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["configurable"]
    deps = cfg["deps"]
    data = Path(state["file_path"]).read_bytes()
    text = extract_text(data, mime_type=state["mime_type"], vision=deps.vision,
                        progress=cfg.get("progress"))
    return {"ocr_text": text, "content_hash": hashlib.sha256(data).hexdigest()}


def dedup_check_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Hash the staged file BEFORE OCR. If this exact file was ingested already,
    short-circuit: reuse the existing document and skip OCR/extraction/HITL."""
    deps = config["configurable"]["deps"]
    data = Path(state["file_path"]).read_bytes()
    chash = hashlib.sha256(data).hexdigest()
    with deps.session_factory() as s:
        existing = find_by_content_hash(s, chash)
        dup = (existing.id, existing.patient_id) if existing is not None else None
    if dup is not None:
        staged = state.get("file_path")
        if staged and os.path.exists(staged):
            try:
                os.remove(staged)
            except OSError:
                pass
        return {"dedup": "duplicate", "already_ingested": True,
                "content_hash": chash, "document_id": dup[0], "patient_id": dup[1],
                "messages": state["messages"] + [{
                    "role": "assistant",
                    "content": "This document was already on file — skipped to avoid duplicates.",
                }]}
    return {"dedup": "new", "content_hash": chash}


def extract_entities_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    text = state["ocr_text"]

    # Cache the structured extraction by (model, document text): re-ingesting the
    # same document skips the slow LLM call. Stored as the model_dump() dict.
    key = make_key(f"extract:{get_settings().ollama_model}", text)
    extracted = get_or_set(
        key,
        lambda: deps.chat.structured(
            _EXTRACT_PROMPT.format(text=text), ExtractionResult
        ).model_dump(),
    )
    return {"extracted": extracted}


def resolve_patient_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Match the extracted name to existing patients. An exact (case-insensitive) single
    match auto-resolves. A close-but-not-exact name becomes a candidate so the confirm gate
    can ask 'same person as X?' — never silently creating a near-duplicate profile."""
    deps = config["configurable"]["deps"]
    name = (state.get("extracted") or {}).get("patient_name")
    if not name:
        return {"patient_id": None, "patient_candidates": []}
    nlow = name.strip().lower()
    with deps.session_factory() as s:
        all_p = s.query(Patient).all()
        exact = [{"id": p.id, "name": p.name} for p in all_p
                 if p.name.strip().lower() == nlow]
        fuzzy = [{"id": p.id, "name": p.name} for p in all_p
                 if p.name.strip().lower() != nlow
                 and SequenceMatcher(None, nlow, p.name.strip().lower()).ratio() >= 0.85]
    if len(exact) == 1:
        return {"patient_id": exact[0]["id"], "patient_candidates": []}
    return {"patient_id": None, "patient_candidates": exact + fuzzy}


def confirm_ingest_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Single HITL gate: human reviews the patient AND the extracted entities in one
    step. The same approval selects/creates the patient and commits the entities."""
    deps = config["configurable"]["deps"]
    ex = state.get("extracted") or {}
    decision = interrupt({
        "type": "confirm_ingest",
        "extracted": ex,
        "candidates": state.get("patient_candidates", []),
        "patient_id": state.get("patient_id"),  # set if a single existing match
        "extracted_name": ex.get("patient_name"),
        "extracted_age": ex.get("patient_age"),
        "extracted_gender": ex.get("patient_gender"),
    })
    if not decision.get("approved"):
        return {"extracted": None, "intent": "rejected"}

    extracted = decision.get("extracted", ex)
    # Patient: use the chosen/pre-matched id, else create a new profile.
    pid = decision.get("patient_id") or state.get("patient_id")
    if pid is None:
        with deps.session_factory() as s:
            p = create_patient(
                s,
                name=decision.get("name") or extracted.get("patient_name") or "Unknown",
                age=decision.get("age", extracted.get("patient_age")),
                gender=decision.get("gender") or extracted.get("patient_gender"),
            )
            pid = p.id
    return {"extracted": extracted, "patient_id": int(pid)}


def create_document_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Create the document row now that the patient is known, then move the staged
    file into the patient-scoped path. (Document creation is deferred to here so the
    agent — not the user — determines the patient.)"""
    deps = config["configurable"]["deps"]
    if state.get("document_id"):
        return {}  # document already exists (e.g. re-ingest of an existing doc)
    ex = state.get("extracted") or {}
    ext = state.get("file_ext") or "bin"
    staged = state.get("file_path")
    chash = state.get("content_hash")

    # Idempotency: if this exact file was already ingested, reuse that document
    # and skip persist/index so the same upload can't duplicate entities.
    if chash:
        with deps.session_factory() as s:
            existing = find_by_content_hash(s, chash)
            dup = (existing.id, existing.patient_id) if existing is not None else None
        if dup is not None:
            if staged and os.path.exists(staged):
                try:
                    os.remove(staged)
                except OSError:
                    pass
            return {"document_id": dup[0], "patient_id": dup[1],
                    "already_ingested": True}

    with deps.session_factory() as s:
        doc = create_document(
            s, patient_id=state["patient_id"], doc_type=ex.get("doc_type"),
            source_type=state.get("source_type"), mime_type=state.get("mime_type"),
            content_hash=chash,
            report_date=parse_doc_date(ex.get("doc_date")),
            original_name=state.get("original_name"),
        )
        doc_id = doc.id
        final_path = staged
        if staged and os.path.exists(staged):
            data = Path(staged).read_bytes()
            final_path = storage.save_bytes(state["patient_id"], doc_id, ext, data)
            set_file_path(s, doc_id, final_path)
            try:
                os.remove(staged)
            except OSError:
                pass
    return {"document_id": doc_id, "file_path": final_path}


def persist_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if state.get("already_ingested"):
        return {"messages": state["messages"] + [{
            "role": "assistant",
            "content": "This document was already on file — skipped to avoid duplicates.",
        }]}
    deps = config["configurable"]["deps"]
    result = ExtractionResult(**state["extracted"])
    with deps.session_factory() as s:
        n = persist_extraction(s, document_id=state["document_id"], result=result)
    pname = (state.get("extracted") or {}).get("patient_name") or "patient"
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Saved {n} entities for {pname}."}
    ]}


def chunk_embed_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if state.get("already_ingested"):
        return {}  # already indexed on the first ingest
    deps = config["configurable"]["deps"]
    ex = state["extracted"]
    header = f"{ex.get('patient_name') or ''} · {ex.get('doc_type') or 'doc'} · {ex.get('doc_date') or ''}".strip()
    with deps.session_factory() as s:
        doc = get_document(s, state["document_id"])
        if doc is not None and state.get("ocr_text"):
            doc.raw_ocr_text = state["ocr_text"]
            doc.status = "indexed"
            s.commit()
        text = state.get("ocr_text") or ""
        chunks = make_semantic_chunks(text, deps.embedder, header=header) if text else []
        n = chunk_and_embed(
            s, document_id=state["document_id"], patient_id=state["patient_id"],
            text=text, header=header, embedder=deps.embedder, chunks=chunks or None,
        )
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Indexed {n} chunks. Ingestion complete."}
    ]}
