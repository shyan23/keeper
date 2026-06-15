from app.models import (
    Patient, Document, Doctor, Disease, Symptom, Medication,
    MedicalTest, TestResult, DocumentEntity, Chunk,
)
from app.db import Base


def test_tables_registered():
    expected = {
        "patient", "document", "doctor", "disease", "symptom",
        "medication", "medical_test", "test_result",
        "document_entity", "chunk",
    }
    assert expected.issubset(set(Base.metadata.tables.keys()))


def test_document_has_file_path():
    cols = {c.name for c in Document.__table__.columns}
    assert "file_path" in cols
    assert "raw_ocr_text" in cols


def test_chunk_has_vector_and_patient():
    cols = {c.name for c in Chunk.__table__.columns}
    assert {"embedding", "patient_id", "document_id"} <= cols


def test_patient_columns():
    cols = {c.name for c in Patient.__table__.columns}
    assert {"id", "name", "age", "gender", "relationship", "created_at"} <= cols
