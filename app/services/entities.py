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
