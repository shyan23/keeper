from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session

from app.db import SessionLocal
from app.services import graph as gsvc
from app.services import patients as psvc

router = APIRouter(prefix="/api")


@router.get("/patients/{patient_id}/graph")
def patient_graph(patient_id: int) -> dict:
    """Returns the full medical relationship graph for a patient."""
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise HTTPException(status_code=404, detail="patient not found")
        return gsvc.build_graph(db, patient_id)
    finally:
        db.close()
