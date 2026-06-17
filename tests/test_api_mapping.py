from app.api.schemas import (
    DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut,
)


def test_patient_out_serializes():
    p = PatientOut(id="1", name="Jane", age=42, gender="female",
                   bloodType="—", image="http://x", lastVisit="2026-06-01",
                   status="Active")
    assert p.model_dump()["id"] == "1"


def test_patient_in_optional_fields():
    p = PatientIn(name="Jane")
    assert p.age is None and p.gender is None and p.relationship is None
