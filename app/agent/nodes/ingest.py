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
