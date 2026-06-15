from sqlalchemy.orm import Session

from app.models import Patient


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


def delete_patient(db: Session, patient_id: int) -> None:
    patient = db.get(Patient, patient_id)
    if patient is not None:
        db.delete(patient)
        db.commit()
