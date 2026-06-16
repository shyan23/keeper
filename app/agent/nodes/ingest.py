from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from app import storage
from app.agent.state import ExtractionResult
from app.cache import get_or_set, make_key
from app.config import get_settings
from app.services.chunking import chunk_and_embed, make_semantic_chunks
from app.services.documents import create_document, get_document, set_file_path
from app.services.entities import persist_extraction
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
    return {"ocr_text": text}


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


def confirm_entities_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate: human reviews/edits extracted entities; same approval commits the write."""
    decision = interrupt({"type": "confirm_entities", "extracted": state["extracted"]})
    if not decision.get("approved"):
        return {"extracted": None, "intent": "rejected"}
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
    """HITL gate (reached when patient is new/ambiguous): pick an existing patient
    or create a new profile from the extracted name/age/gender (human-verified)."""
    if state.get("patient_id"):
        return {}
    deps = config["configurable"]["deps"]
    ex = state.get("extracted") or {}
    decision = interrupt({
        "type": "confirm_patient",
        "candidates": state.get("patient_candidates", []),
        "extracted_name": ex.get("patient_name"),
        "extracted_age": ex.get("patient_age"),
        "extracted_gender": ex.get("patient_gender"),
    })
    pid = decision.get("patient_id")
    if pid is None and decision.get("create_new"):
        with deps.session_factory() as s:
            p = create_patient(
                s,
                name=decision.get("name") or ex.get("patient_name") or "Unknown",
                age=decision.get("age", ex.get("patient_age")),
                gender=decision.get("gender") or ex.get("patient_gender"),
            )
            pid = p.id
    return {"patient_id": pid}


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
    with deps.session_factory() as s:
        doc = create_document(
            s, patient_id=state["patient_id"], doc_type=ex.get("doc_type"),
            source_type=state.get("source_type"), mime_type=state.get("mime_type"),
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
    deps = config["configurable"]["deps"]
    result = ExtractionResult(**state["extracted"])
    with deps.session_factory() as s:
        n = persist_extraction(s, document_id=state["document_id"], result=result)
    pname = (state.get("extracted") or {}).get("patient_name") or "patient"
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Saved {n} entities for {pname}."}
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
        text = state.get("ocr_text") or ""
        chunks = make_semantic_chunks(text, deps.embedder, header=header) if text else []
        n = chunk_and_embed(
            s, document_id=state["document_id"], patient_id=state["patient_id"],
            text=text, header=header, embedder=deps.embedder, chunks=chunks or None,
        )
    return {"messages": state["messages"] + [
        {"role": "assistant", "content": f"Indexed {n} chunks. Ingestion complete."}
    ]}
