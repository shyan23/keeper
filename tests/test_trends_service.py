import datetime as dt

from app.db import SessionLocal
from app.models import (
    Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services import trends


def _add_result(db, patient, doc_date, test_name, value, unit, ref):
    doc = Document(patient_id=patient.id, doc_type="lab report",
                   report_date=doc_date)
    db.add(doc); db.flush()
    mt = MedicalTest(name=test_name)
    db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value=value, unit=unit,
                    reference_range=ref)
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                          entity_id=tr.id))
    db.commit()


def test_list_metrics_needs_two_numeric_points():
    db = SessionLocal()
    try:
        p = Patient(name="Trend One"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2024, 1, 1), "Hemoglobin", "12", "g/dL", "12-16")
        _add_result(db, p, dt.date(2025, 1, 1), "Hemoglobin", "11", "g/dL", "12-16")
        _add_result(db, p, dt.date(2025, 1, 1), "Glucose", "90", "mg/dL", "70-100")  # only 1 point
        metrics = trends.list_metrics(db, p.id)
        keys = {m["key"] for m in metrics}
        assert "hemoglobin" in keys
        assert "glucose" not in keys
        hgb = next(m for m in metrics if m["key"] == "hemoglobin")
        assert hgb["label"] == "Hemoglobin"
        assert hgb["unit"] == "g/dL"
        assert hgb["n"] == 2
    finally:
        db.close()


def test_metric_series_sorted_with_range_flags():
    db = SessionLocal()
    try:
        p = Patient(name="Trend Two"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2025, 1, 1), "Hemoglobin", "11", "g/dL", "12-16")  # below
        _add_result(db, p, dt.date(2024, 1, 1), "Hemoglobin", "14", "g/dL", "12-16")  # in
        s = trends.metric_series(db, p.id, "hemoglobin")
        assert s["ref_low"] == 12.0 and s["ref_high"] == 16.0
        # sorted ascending by date
        assert [pt["date"] for pt in s["points"]] == ["2024-01-01", "2025-01-01"]
        assert [pt["value"] for pt in s["points"]] == [14.0, 11.0]
        assert [pt["in_range"] for pt in s["points"]] == [True, False]
    finally:
        db.close()


def test_metric_series_skips_non_numeric_and_undated():
    db = SessionLocal()
    try:
        p = Patient(name="Trend Three"); db.add(p); db.flush()
        _add_result(db, p, dt.date(2025, 1, 1), "Culture", "Positive", "", "")
        _add_result(db, p, None, "Culture", "5", "", "")  # undated -> excluded
        s = trends.metric_series(db, p.id, "culture")
        assert s["points"] == []
        assert s["ref_low"] is None and s["ref_high"] is None
    finally:
        db.close()
