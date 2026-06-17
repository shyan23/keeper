"""Locate and apply a single, human-confirmed correction to extracted data.

The agent parses a natural-language edit request into an `EditPlan`, this module
finds the matching record (newest document first, scoped to one patient), and —
only after the human confirms — applies the change. Nothing here commits without
an explicit apply call from the confirm gate.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Callable

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models import (
    Disease, Document, DocumentEntity, MedicalTest, Medication, Symptom, TestResult,
)
from app.services.dates import parse_doc_date

# field -> TestResult column
_TEST_FIELDS = {"test_value": "value", "test_unit": "unit", "test_reference": "reference_range"}
# field -> normalized-name model + DocumentEntity.entity_type
_NAME_MODELS = {
    "disease": (Disease, "disease"),
    "symptom": (Symptom, "symptom"),
    "medication": (Medication, "medication"),
}
_FIELD_LABEL = {
    "test_value": "value", "test_unit": "unit", "test_reference": "reference range",
    "disease": "diagnosis", "symptom": "symptom", "medication": "medication",
    "doc_type": "document type", "report_date": "document date",
}


def _doc_order(q):
    return q.order_by(
        func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(),
        Document.id.desc(),
    )


# Filler words that shouldn't affect a test/entity name match.
_FILLER = {"level", "levels", "count", "counts", "value", "values", "reading", "readings",
           "result", "results", "test", "the", "of", "in", "a", "an", "percentage", "percent"}
_MATCH_THRESHOLD = 0.5


def _norm_tokens(s: str) -> list[str]:
    """Lowercase, fold common British/American medical spellings (haemo->hemo,
    anaemia->anemia), drop punctuation and filler words. So 'haemoglobin level'
    and 'Hemoglobin (Hb%)' overlap."""
    s = s.lower().replace("haemo", "hemo").replace("oe", "e").replace("ae", "e")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return [t for t in s.split() if t and t not in _FILLER]


def _match_score(target: str, name: str) -> float:
    tt, nt = _norm_tokens(target), _norm_tokens(name)
    if not tt or not nt:
        return 0.0
    ts, ns = set(tt), set(nt)
    overlap = len(ts & ns) / len(ts) if ts else 0.0   # how much of the target is present
    seq = SequenceMatcher(None, " ".join(tt), " ".join(nt)).ratio()
    return max(overlap, seq)


def _select(rows: list, target: str, name_of: Callable[[Any], str]):
    """From rows (already newest-document-first) pick the best fuzzy name match.
    Empty target -> newest row. Stable sort keeps the latest among equal scores."""
    if not rows:
        return None
    if not target:
        return rows[0]
    scored = [(r, _match_score(target, name_of(r))) for r in rows]
    scored = [(r, sc) for r, sc in scored if sc >= _MATCH_THRESHOLD]
    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0]


def _doc_info(doc: Document) -> dict[str, Any]:
    return {
        "document_id": doc.id,
        "doc_type": doc.doc_type or "document",
        "date": (doc.report_date.strftime("%Y-%m-%d") if doc.report_date
                 else doc.uploaded_at.strftime("%Y-%m-%d") if doc.uploaded_at else None),
        "name": doc.original_name or f"document-{doc.id}",
    }


def find_edit_target(db: Session, patient_id: int, plan: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve an EditPlan to a concrete record. Returns a proposal dict (current +
    proposed values, document context, and the ids needed to apply) or None."""
    field = (plan.get("field") or "").strip()
    name = (plan.get("target_name") or "").strip()
    new_value = (plan.get("new_value") or "").strip()
    doc_type_hint = (plan.get("doc_type") or "").strip()

    if field in _TEST_FIELDS:
        col = _TEST_FIELDS[field]
        q = (db.query(TestResult, MedicalTest.name, Document)
             .join(MedicalTest, MedicalTest.id == TestResult.medical_test_id)
             .join(DocumentEntity, and_(DocumentEntity.entity_id == TestResult.id,
                                        DocumentEntity.entity_type == "test_result"))
             .join(Document, Document.id == DocumentEntity.document_id)
             .filter(Document.patient_id == patient_id))
        row = _select(_doc_order(q).all(), name, name_of=lambda r: r[1])
        if not row:
            return None
        tr, tname, doc = row
        return {
            "kind": "test", "field": field, "ref_id": tr.id,
            "subject": tname, "field_label": _FIELD_LABEL[field],
            "label": f"{tname} ({_FIELD_LABEL[field]})",
            "current": getattr(tr, col) or "", "proposed": new_value,
            **_doc_info(doc),
        }

    if field in _NAME_MODELS:
        model, etype = _NAME_MODELS[field]
        q = (db.query(model, Document)
             .join(DocumentEntity, and_(DocumentEntity.entity_id == model.id,
                                        DocumentEntity.entity_type == etype))
             .join(Document, Document.id == DocumentEntity.document_id)
             .filter(Document.patient_id == patient_id))
        row = _select(_doc_order(q).all(), name, name_of=lambda r: r[0].name)
        if not row:
            return None
        obj, doc = row
        return {
            "kind": "name", "field": field, "ref_id": obj.id,
            "subject": obj.name, "field_label": _FIELD_LABEL[field],
            "label": f"{_FIELD_LABEL[field]} “{obj.name}”",
            "current": obj.name, "proposed": new_value,
            **_doc_info(doc),
        }

    if field in ("doc_type", "report_date"):
        q = db.query(Document).filter(Document.patient_id == patient_id)
        if doc_type_hint:
            q = q.filter(func.lower(Document.doc_type).like(f"%{doc_type_hint.lower()}%"))
        doc = _doc_order(q).first()
        if not doc:
            return None
        current = (doc.doc_type or "") if field == "doc_type" else (
            doc.report_date.strftime("%Y-%m-%d") if doc.report_date else "")
        return {
            "kind": "document", "field": field, "ref_id": doc.id,
            "subject": doc.original_name or f"document-{doc.id}",
            "field_label": _FIELD_LABEL[field],
            "label": f"{_FIELD_LABEL[field]}", "current": current, "proposed": new_value,
            **_doc_info(doc),
        }
    return None


def apply_edit(db: Session, target: dict[str, Any]) -> None:
    """Commit a single confirmed edit. `target` is a proposal from find_edit_target,
    optionally with `proposed` overridden by the human in the confirm card."""
    proposed = (target.get("proposed") or "").strip()
    kind, field, ref_id = target["kind"], target["field"], target["ref_id"]
    if kind == "test":
        tr = db.get(TestResult, ref_id)
        if tr is not None:
            setattr(tr, _TEST_FIELDS[field], proposed or None)
    elif kind == "name":
        model = _NAME_MODELS[field][0]
        obj = db.get(model, ref_id)
        if obj is not None:
            obj.name = proposed
    elif kind == "document":
        doc = db.get(Document, ref_id)
        if doc is not None:
            if field == "doc_type":
                doc.doc_type = proposed or None
            else:
                doc.report_date = parse_doc_date(proposed)
    db.commit()
