from __future__ import annotations

from pydantic import BaseModel


class HealthOut(BaseModel):
    status: str
    db: str
    pgvector: bool
    version: str


class PatientOut(BaseModel):
    id: str
    name: str
    age: int | None = None
    gender: str | None = None
    bloodType: str = "—"
    image: str
    lastVisit: str
    status: str = "Active"


class PatientIn(BaseModel):
    name: str
    age: int | None = None
    gender: str | None = None
    relationship: str | None = None


class RecordOut(BaseModel):
    id: str
    documentId: str = ""
    patientId: str
    type: str  # disease | symptom | medicine | test_result | treatment_plan
    title: str
    description: str
    value: str = ""
    unit: str = ""
    reference: str = ""
    date: str | None = None
    status: str = "Recorded"
    severity: str | None = None
    doctor: str | None = None


class DocumentOut(BaseModel):
    id: str
    name: str
    date: str | None = None
    type: str
    size: str
    category: str | None = None


class DeleteRecordsIn(BaseModel):
    document_ids: list[str]
