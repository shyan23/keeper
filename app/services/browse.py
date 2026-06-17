"""Read-side queries for the organization/browse pages.

Each row joins an extracted entity back through `document_entity` -> `document`
-> `patient`, so the UI can show what was found, for whom, from which document
(date), and the source span that proves it.
"""
from __future__ import annotations

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from app.models import (
    Disease, Doctor, Document, DocumentEntity, MedicalTest, Medication,
    Patient, Symptom, TestResult,
)

# entity_type -> the normalized name table it points at
_NAME_MODELS = {
    "disease": Disease,
    "symptom": Symptom,
    "medication": Medication,
    "doctor": Doctor,
}


def list_entity_links(db: Session, entity_type: str, patient_id: int | None = None) -> list[dict]:
    """Name-based entities (disease/symptom/medication/doctor) with their provenance."""
    model = _NAME_MODELS[entity_type]
    q = (
        db.query(
            model.name, Patient.name.label("patient"), Patient.id.label("patient_id"),
            Document.doc_type, Document.uploaded_at,
            DocumentEntity.confidence, DocumentEntity.source_span,
            DocumentEntity.document_id, Document.report_date,
        )
        .join(DocumentEntity, and_(DocumentEntity.entity_id == model.id,
                                   DocumentEntity.entity_type == entity_type))
        .join(Document, Document.id == DocumentEntity.document_id)
        .join(Patient, Patient.id == Document.patient_id)
    )
    if patient_id is not None:
        q = q.filter(Patient.id == patient_id)
    q = q.order_by(func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(), model.name)
    return [
        {"name": r[0], "patient": r[1], "patient_id": r[2], "doc_type": r[3],
         "date": (r[8].strftime("%Y-%m-%d") if r[8]
                  else (r[4].strftime("%Y-%m-%d") if r[4] else None)),
         "confidence": round(r[5], 2) if r[5] is not None else None,
         "source": r[6], "document_id": r[7]}
        for r in q.all()
    ]


def list_test_results(db: Session, patient_id: int | None = None) -> list[dict]:
    """Diagnostics: medical tests + their results with provenance."""
    q = (
        db.query(
            MedicalTest.name, TestResult.value, TestResult.unit, TestResult.reference_range,
            Patient.name.label("patient"), Patient.id.label("patient_id"),
            Document.doc_type, Document.uploaded_at, DocumentEntity.source_span,
            DocumentEntity.document_id, Document.report_date,
        )
        .join(TestResult, TestResult.medical_test_id == MedicalTest.id)
        .join(DocumentEntity, and_(DocumentEntity.entity_id == TestResult.id,
                                   DocumentEntity.entity_type == "test_result"))
        .join(Document, Document.id == DocumentEntity.document_id)
        .join(Patient, Patient.id == Document.patient_id)
    )
    if patient_id is not None:
        q = q.filter(Patient.id == patient_id)
    q = q.order_by(func.coalesce(Document.report_date, func.date(Document.uploaded_at)).desc(), MedicalTest.name)
    return [
        {"test": r[0], "value": r[1], "unit": r[2], "reference_range": r[3],
         "patient": r[4], "patient_id": r[5], "doc_type": r[6],
         "date": (r[10].strftime("%Y-%m-%d") if r[10]
                  else (r[7].strftime("%Y-%m-%d") if r[7] else None)),
         "source": r[8], "document_id": r[9]}
        for r in q.all()
    ]


def list_documents_timeline(db: Session, patient_id: int | None = None) -> list[dict]:
    """Documents newest-first, for the date/timeline page."""
    q = db.query(Document, Patient.name).join(Patient, Patient.id == Document.patient_id)
    if patient_id is not None:
        q = q.filter(Patient.id == patient_id)
    q = q.order_by(Document.uploaded_at.desc(), Document.id.desc())
    return [
        {"id": d.id, "patient": pname, "type": d.doc_type, "status": d.status,
         "report_date": d.report_date.strftime("%Y-%m-%d") if d.report_date else None,
         "date": d.uploaded_at.strftime("%Y-%m-%d %H:%M") if d.uploaded_at else None,
         "original_name": d.original_name, "file": d.file_path}
        for d, pname in q.all()
    ]
