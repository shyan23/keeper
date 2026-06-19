from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api import mapping
from app.api.schemas import (
    DeleteRecordsIn, DocumentOut, HealthOut, MetricOut, PatientIn, PatientOut,
    RecordOut, SeriesOut,
)
from app.db import SessionLocal
from app.services import browse as bsvc
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc
from app.services import purge as pgsvc
from app.services import trends as tsvc

router = APIRouter(prefix="/api")


def _last_visit(db: Session, patient_id: int) -> str | None:
    docs = dsvc.list_documents(db, patient_id=patient_id)
    if docs and docs[0].uploaded_at:
        return docs[0].uploaded_at.strftime("%Y-%m-%d")
    p = psvc.get_patient(db, patient_id)
    if p is not None and p.created_at:
        return p.created_at.strftime("%Y-%m-%d")
    return None


@router.get("/health", response_model=HealthOut)
def health() -> dict:
    return hsvc.check_health()


@router.get("/patients", response_model=list[PatientOut])
def list_patients() -> list[dict]:
    db = SessionLocal()
    try:
        return [mapping.patient_to_out(p, _last_visit(db, p.id))
                for p in psvc.list_patients(db)]
    finally:
        db.close()


@router.post("/patients", response_model=PatientOut, status_code=201)
def create_patient(body: PatientIn) -> dict:
    db = SessionLocal()
    try:
        p = psvc.create_patient(db, name=body.name, age=body.age,
                                gender=body.gender, relationship=body.relationship)
        return mapping.patient_to_out(p, _last_visit(db, p.id))
    finally:
        db.close()


@router.get("/patients/{patient_id}/records", response_model=list[RecordOut])
def patient_records(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        diseases = bsvc.list_entity_links(db, "disease", patient_id=patient_id)
        symptoms = bsvc.list_entity_links(db, "symptom", patient_id=patient_id)
        meds = bsvc.list_entity_links(db, "medication", patient_id=patient_id)
        tests = bsvc.list_test_results(db, patient_id=patient_id)
        return mapping.merge_records(str(patient_id), diseases, symptoms, meds, tests)
    finally:
        db.close()


@router.get("/patients/{patient_id}/trends", response_model=list[MetricOut])
def patient_trends(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return tsvc.list_metrics(db, patient_id)
    finally:
        db.close()


@router.get("/patients/{patient_id}/trends/{key}", response_model=SeriesOut)
def patient_trend_series(patient_id: int, key: str) -> dict:
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return tsvc.metric_series(db, patient_id, key)
    finally:
        db.close()


@router.get("/patients/{patient_id}/documents", response_model=list[DocumentOut])
def patient_documents(patient_id: int) -> list[dict]:
    db = SessionLocal()
    try:
        rows = bsvc.list_documents_timeline(db, patient_id=patient_id)
        out = []
        for row in rows:
            size_str = "—"
            path = row.get("file")
            try:
                if path and os.path.exists(path):
                    size_str = f"{os.path.getsize(path) / 1_048_576:.1f} MB"
            except OSError:
                pass
            out.append(mapping.document_to_out(row, size_str))
        return out
    finally:
        db.close()


@router.get("/documents/{document_id}/file")
def document_file(document_id: int):
    """Stream the original uploaded file so citations/docs can open it."""
    db = SessionLocal()
    try:
        doc = dsvc.get_document(db, document_id)
        if doc is None or not doc.file_path or not os.path.exists(doc.file_path):
            raise HTTPException(status_code=404, detail="document file not found")
        filename = doc.original_name or os.path.basename(doc.file_path)
        return FileResponse(doc.file_path,
                            media_type=doc.mime_type or "application/octet-stream",
                            filename=filename)
    finally:
        db.close()


@router.post("/patients/{patient_id}/records/delete")
def delete_records(patient_id: int, body: DeleteRecordsIn) -> dict:
    db = SessionLocal()
    try:
        n = pgsvc.delete_documents(db, patient_id, body.document_ids)
        return {"deleted": n}
    finally:
        db.close()
