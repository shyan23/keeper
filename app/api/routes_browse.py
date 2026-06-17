from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.api import mapping
from app.api.schemas import DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut
from app.db import SessionLocal
from app.services import browse as bsvc
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc

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
