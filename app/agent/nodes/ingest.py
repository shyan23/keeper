from __future__ import annotations

import hashlib
import logging
import os
import re
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
from app.services.dates import date_from_text, parse_doc_date
from app.services.extraction import extract_pages, extract_text, slice_pdf
from app.services.patients import create_patient
from app.services.segment import doc_type_for, split_reports
from app.models import Patient

log = logging.getLogger("app.ingest")


_EXTRACT_PROMPT = """Extract structured medical data from this document text.

patient_name, patient_age, patient_gender, doc_type, doc_date and doctor are PLAIN
values (a string or a number) — NOT objects. patient_age is an integer.
Only the list items in diseases, symptoms, medications and tests are objects with
a name plus confidence (0-1) and source_span (the exact text you used).
If a field is absent, leave it null/empty.

The `tests` list holds BOTH numeric lab results and narrative findings:
- Numeric lab result (CBC, lipid, biochemistry…): name=test, value=the number,
  unit=the unit, reference_range=the normal range.
- Narrative finding (X-ray, ultrasound/USG, CT, MRI, ECG, echo, any imaging or
  descriptive report): make ONE entry per finding line — name=the finding label
  (e.g. "Diaphragm", "Heart", "Lung fields", "Bony thorax", "Impression"),
  value=the finding text verbatim, and leave unit and reference_range null.
  Never collapse a whole imaging report into a single "X-Ray" entry.

Document:
{text}"""


def extract_text_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    cfg = config["configurable"]
    deps = cfg["deps"]
    data = Path(state["file_path"]).read_bytes()
    text = extract_text(data, mime_type=state["mime_type"], vision=deps.vision,
                        progress=cfg.get("progress"))
    # Per-page text (from the OCR cache, no re-OCR) so a multi-report scan can be
    # split into separate documents.
    pages = extract_pages(data, mime_type=state["mime_type"], vision=deps.vision)
    return {"ocr_text": text, "pages": pages,
            "content_hash": hashlib.sha256(data).hexdigest()}


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


def _extract_one(deps, text: str) -> dict[str, Any]:
    # Cache structured extraction by (model, text): re-ingest skips the slow LLM call.
    # v2: prompt now breaks narrative/imaging reports into labeled findings.
    key = make_key(f"extract:v2:{get_settings().ollama_model}", text)
    return get_or_set(
        key,
        lambda: deps.chat.structured(
            _EXTRACT_PROMPT.format(text=text), ExtractionResult
        ).model_dump(),
    )


def _report_name(title: str | None, ex: dict[str, Any]) -> str:
    if title:
        return title
    dt = (ex.get("doc_type") or "").strip()
    return dt.title() if dt else "Medical Report"


def segment_extract_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """LLM-split the scan into reports, then extract each separately, so a bundle of
    5 dated reports becomes 5 documents. Single-report uploads yield one segment.
    `extracted` (first report, with a patient name filled from any segment) drives
    patient resolution and the confirm card."""
    deps = config["configurable"]["deps"]
    pages = state.get("pages") or [state.get("ocr_text") or ""]
    segments: list[dict[str, Any]] = []
    patient_name = None
    for seg in split_reports(deps.chat, pages):
        text = seg["text"]
        ex = _extract_one(deps, text)
        title = seg.get("title")
        # Prefer the report's own date: LLM split date, else LLM extraction
        # doc_date, else scraped from text.
        rdate = (parse_doc_date(seg.get("date")) or parse_doc_date(ex.get("doc_date"))
                 or date_from_text(text))
        # Topic name + category come from the LLM split; fall back to extraction.
        doc_type = (seg.get("doc_type") if seg.get("doc_type") not in (None, "", "document")
                    else (ex.get("doc_type") or doc_type_for(title)))
        segments.append({
            "name": title or _report_name(None, ex),
            "doc_type": doc_type or "document",
            "report_date": rdate.isoformat() if rdate else None,
            "extracted": ex,
            "text": text,
            "pages": seg.get("pages") or [],   # source page indices -> slice file per report
        })
        if not patient_name:
            patient_name = ex.get("patient_name")
    first = dict(segments[0]["extracted"])
    if patient_name and not first.get("patient_name"):
        first["patient_name"] = patient_name
    return {"segments": segments, "extracted": first}


# Honorifics / titles that should NOT distinguish two patients.
_TITLE_TOKENS = {
    "mr", "mrs", "ms", "miss", "mst", "master", "md", "dr", "prof", "professor",
    "mister", "sir", "madam", "smt", "mr.", "mrs.", "dr.",
}


def _normalize_name(name: str | None) -> str:
    """Lowercase, drop honorifics/titles and punctuation, collapse whitespace so
    'MRS. NAFISA KABIR' and 'Nafisa Kabir' compare as the same person."""
    n = re.sub(r"[.\,]", " ", (name or "").lower())
    toks = [t for t in n.split() if t and t not in _TITLE_TOKENS]
    return " ".join(toks)


def resolve_patient_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Match the extracted name to existing patients. A single normalized match
    auto-resolves. A close-but-not-exact name (or an ambiguous tie) becomes a
    candidate so the confirm gate can ask 'same person as X?' — never silently
    creating a near-duplicate profile. Honorifics ('MRS.', 'Dr.') are ignored."""
    deps = config["configurable"]["deps"]
    name = (state.get("extracted") or {}).get("patient_name")
    if not name:
        return {"patient_id": None, "patient_candidates": []}
    nnorm = _normalize_name(name)
    with deps.session_factory() as s:
        all_p = s.query(Patient).all()
        exact = [{"id": p.id, "name": p.name} for p in all_p
                 if _normalize_name(p.name) == nnorm]
        fuzzy = [{"id": p.id, "name": p.name} for p in all_p
                 if _normalize_name(p.name) != nnorm
                 and SequenceMatcher(None, nnorm, _normalize_name(p.name)).ratio() >= 0.85]
    if len(exact) == 1:
        return {"patient_id": exact[0]["id"], "patient_candidates": []}
    # 0 matches -> brand new; 2+ normalized matches -> ambiguous, let the human pick.
    return {"patient_id": None, "patient_candidates": exact + fuzzy}


def confirm_ingest_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Single HITL gate: human reviews the patient AND the extracted entities in one
    step. The same approval selects/creates the patient and commits the entities."""
    deps = config["configurable"]["deps"]
    ex = state.get("extracted") or {}
    segments = state.get("segments") or []
    # Every detected report is sent for review/edit, each with its own title, date
    # and entities, so the human verifies all N reports (not just the first).
    seg_payload = [{
        "name": s.get("name"), "doc_type": s.get("doc_type"),
        "date": s.get("report_date"), "extracted": s.get("extracted") or {},
    } for s in segments]
    decision = interrupt({
        "type": "confirm_ingest",
        "extracted": ex,            # first report (back-compat)
        "segments": seg_payload,    # all reports, editable one by one
        "candidates": state.get("patient_candidates", []),
        "patient_id": state.get("patient_id"),  # set if a single existing match
        "extracted_name": ex.get("patient_name"),
        "extracted_age": ex.get("patient_age"),
        "extracted_gender": ex.get("patient_gender"),
    })
    if not decision.get("approved"):
        return {"extracted": None, "intent": "rejected"}

    # Merge per-report human edits back onto each segment.
    edited = decision.get("segments")
    if edited:
        merged = []
        for i, s in enumerate(segments):
            e = edited[i] if i < len(edited) else {}
            merged.append({
                **s,
                "extracted": e.get("extracted", s["extracted"]),
                "name": e.get("name") or s.get("name"),
                "doc_type": e.get("doc_type") or s.get("doc_type"),
                "report_date": e.get("date") if e.get("date") is not None else s.get("report_date"),
            })
        segments = merged
    elif segments:  # back-compat: single edited blob
        segments = [{**segments[0], "extracted": decision.get("extracted", ex)}] + segments[1:]
    extracted = (segments[0]["extracted"] if segments else decision.get("extracted", ex))
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
    return {"extracted": extracted, "segments": segments, "patient_id": int(pid)}


def persist_reports_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Create one document PER detected report (own date, topic name, entities,
    chunks). All share the original file so each card opens the same PDF. Patient is
    already resolved by the confirm gate."""
    if state.get("already_ingested"):
        return {"messages": state["messages"] + [{
            "role": "assistant",
            "content": "This document was already on file — skipped to avoid duplicates.",
        }]}
    deps = config["configurable"]["deps"]
    progress = config["configurable"].get("progress")
    pid = state["patient_id"]
    ext = state.get("file_ext") or "bin"
    staged = state.get("file_path")
    chash = state.get("content_hash")
    segments = state.get("segments") or [{
        "name": state.get("original_name"),
        "doc_type": (state.get("extracted") or {}).get("doc_type") or "document",
        "report_date": None, "extracted": state.get("extracted") or {},
        "text": state.get("ocr_text") or "",
    }]
    data = (Path(staged).read_bytes() if staged and os.path.exists(staged) else None)

    titles: list[str] = []
    total_entities = 0
    total_chunks = 0
    with deps.session_factory() as s:
        for seg in segments:
            ex = seg["extracted"]
            result = ExtractionResult(**ex)
            rdate = (parse_doc_date(seg.get("report_date"))
                     or parse_doc_date(ex.get("doc_date"))
                     or date_from_text(seg.get("text")))
            doc = create_document(
                s, patient_id=pid, doc_type=seg.get("doc_type") or ex.get("doc_type"),
                source_type=state.get("source_type"), mime_type=state.get("mime_type"),
                content_hash=chash, report_date=rdate,
                original_name=seg.get("name") or state.get("original_name"),
            )
            doc_id = doc.id
            if data is not None:
                # Save only THIS report's pages so its card opens just that report,
                # not the whole bundle. Multi-report PDFs get sliced; everything else
                # (single report, images) saves the original bytes.
                blob = data
                if (len(segments) > 1 and state.get("mime_type") == "application/pdf"
                        and seg.get("pages")):
                    blob = slice_pdf(data, seg["pages"])
                    # slice_pdf returns the SAME object when it kept every page —
                    # a degenerate split, so this report's card opens the whole
                    # bundle. Surface it instead of silently saving the full file.
                    if blob is data:
                        name = seg.get("name") or seg.get("doc_type") or "report"
                        warn = (f"⚠ '{name}' kept all pages (pages={seg['pages']}) — "
                                "card will open the whole bundle, not just this report")
                        log.warning("[ingest] %s", warn)
                        if progress:
                            progress(warn)
                set_file_path(s, doc_id, storage.save_bytes(pid, doc_id, ext, blob))
            total_entities += persist_extraction(s, document_id=doc_id, result=result)
            text = seg.get("text") or ""
            header = (f"{ex.get('patient_name') or ''} · {seg.get('name') or 'doc'} · "
                      f"{seg.get('report_date') or ''}").strip()
            d2 = get_document(s, doc_id)
            if d2 is not None and text:
                d2.raw_ocr_text = text
                d2.status = "indexed"
                s.commit()
            chunks = make_semantic_chunks(text, deps.embedder, header=header) if text else []
            total_chunks += chunk_and_embed(
                s, document_id=doc_id, patient_id=pid, text=text, header=header,
                embedder=deps.embedder, chunks=chunks or None,
            )
            titles.append(seg.get("name") or "report")

    if staged and os.path.exists(staged):
        try:
            os.remove(staged)
        except OSError:
            pass

    if len(titles) > 1:
        msg = (f"Split into {len(titles)} reports — {', '.join(titles)}. "
               f"Saved {total_entities} entities, indexed {total_chunks} chunks.")
    else:
        msg = f"Saved {total_entities} entities, indexed {total_chunks} chunks. Ingestion complete."
    return {"messages": state["messages"] + [{"role": "assistant", "content": msg}]}
