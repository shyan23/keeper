import datetime as dt

from app.services import report


def test_resolve_timeframe_explicit_years():
    req = {"years": [2021, 2022]}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2021, 1, 1)
    assert hi == dt.date(2022, 12, 31)


def test_resolve_timeframe_last_n_years():
    req = {"last_n_years": 3}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2023, 6, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_last_n_months():
    req = {"last_n_months": 4}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 6, 19))
    assert lo == dt.date(2026, 2, 19)
    assert hi == dt.date(2026, 6, 19)


def test_resolve_timeframe_months_cross_year():
    req = {"last_n_months": 8}
    lo, hi = report.resolve_timeframe(req, dt.date(2026, 3, 10))
    assert lo == dt.date(2025, 7, 10)
    assert hi == dt.date(2026, 3, 10)


def test_resolve_timeframe_none_is_all_time():
    assert report.resolve_timeframe({}, dt.date(2026, 6, 19)) == (None, None)


def test_resolve_timeframe_leap_day_guard():
    lo, hi = report.resolve_timeframe({"last_n_years": 1}, dt.date(2024, 2, 29))
    assert lo == dt.date(2023, 2, 28)


from app.db import SessionLocal
from app.models import (
    Disease, Document, DocumentEntity, MedicalTest, Patient, TestResult,
)


def _doc(db, patient, doc_date, doc_type, raw=""):
    d = Document(patient_id=patient.id, doc_type=doc_type, report_date=doc_date,
                 raw_ocr_text=raw, original_name=f"{doc_type}.pdf")
    db.add(d); db.flush()
    return d


def _disease(db, doc, name):
    dis = Disease(name=name); db.add(dis); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="disease", entity_id=dis.id))


def _result(db, doc, test_name, value, unit, ref):
    mt = MedicalTest(name=test_name); db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value=value, unit=unit, reference_range=ref)
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result", entity_id=tr.id))


def test_gather_filters_to_window_and_doc_type():
    db = SessionLocal()
    try:
        p = Patient(name="Gather One", age=40); db.add(p); db.flush()
        d_in = _doc(db, p, dt.date(2022, 5, 1), "lipid profile", raw="Age: 55 years")
        _result(db, d_in, "LDL", "130", "mg/dL", "0-100")
        d_old = _doc(db, p, dt.date(2018, 1, 1), "lipid profile")
        _result(db, d_old, "LDL", "90", "mg/dL", "0-100")
        d_other = _doc(db, p, dt.date(2022, 6, 1), "x-ray")
        db.commit()
        data = report.gather(db, p.id, ["lipid profile"],
                             dt.date(2021, 1, 1), dt.date(2023, 1, 1))
        names = {doc["original_name"] for doc in data["documents"]}
        assert names == {"lipid profile.pdf"}
        assert any(t["test"] == "LDL" and t["value"] == "130" for t in data["tests"])
        assert all(t["value"] != "90" for t in data["tests"])
    finally:
        db.close()


def test_gather_most_recent_age_from_newest_doc_ocr():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Age", age=40); db.add(p); db.flush()
        _doc(db, p, dt.date(2020, 1, 1), "lab report", raw="Age: 50 years")
        _doc(db, p, dt.date(2023, 1, 1), "lab report", raw="Age : 53 yrs")
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert data["age"] == 53
    finally:
        db.close()


def test_gather_age_falls_back_to_patient_age():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Fallback", age=72); db.add(p); db.flush()
        _doc(db, p, dt.date(2023, 1, 1), "lab report", raw="no age here")
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert data["age"] == 72
    finally:
        db.close()


def test_gather_dedupes_diseases_preserving_order():
    db = SessionLocal()
    try:
        p = Patient(name="Gather Dx"); db.add(p); db.flush()
        d1 = _doc(db, p, dt.date(2021, 1, 1), "note"); _disease(db, d1, "Anemia")
        d2 = _doc(db, p, dt.date(2022, 1, 1), "note"); _disease(db, d2, "Anemia")
        d3 = _doc(db, p, dt.date(2022, 6, 1), "note"); _disease(db, d3, "Diabetes")
        db.commit()
        data = report.gather(db, p.id, [], None, None)
        assert [x["name"] for x in data["diseases"]] == ["Anemia", "Diabetes"]
    finally:
        db.close()
