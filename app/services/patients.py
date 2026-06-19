from datetime import date

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Document, Patient


def create_patient(db: Session, *, name: str, age: int | None = None,
                   gender: str | None = None, relationship: str | None = None) -> Patient:
    patient = Patient(name=name, age=age, gender=gender, relationship=relationship)
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient


def list_patients(db: Session) -> list[Patient]:
    return db.query(Patient).order_by(Patient.id).all()


def get_patient(db: Session, patient_id: int) -> Patient | None:
    return db.get(Patient, patient_id)


def update_patient(db: Session, patient_id: int, **fields) -> Patient:
    patient = db.get(Patient, patient_id)
    if patient is None:
        raise ValueError(f"patient {patient_id} not found")
    for key, value in fields.items():
        if value is not None:
            setattr(patient, key, value)
    db.commit()
    db.refresh(patient)
    return patient


def apply_latest_demographics(db: Session, patient_id: int, *, age: int | None = None,
                              gender: str | None = None, blood_type: str | None = None,
                              doc_date: date | None = None) -> None:
    """Track header demographics from an ingested document. A field is updated when
    this document is the newest on file (so an older report uploaded later can't roll
    the age back), OR when the patient still has no value for it. None values are
    ignored, so a document that omits blood won't blank an earlier one."""
    patient = db.get(Patient, patient_id)
    if patient is None:
        return
    latest = (db.query(func.max(Document.report_date))
              .filter(Document.patient_id == patient_id).scalar())
    is_newest = doc_date is not None and (latest is None or doc_date >= latest)
    changed = False
    for field, value in (("age", age), ("gender", gender), ("blood_type", blood_type)):
        if value is None:
            continue
        if is_newest or getattr(patient, field) is None:
            setattr(patient, field, value)
            changed = True
    if changed:
        db.commit()


def delete_patient(db: Session, patient_id: int) -> None:
    patient = db.get(Patient, patient_id)
    if patient is not None:
        db.delete(patient)
        db.commit()
