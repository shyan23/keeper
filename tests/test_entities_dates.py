import datetime as dt

from app.agent.state import ExtractedTest, ExtractionResult
from app.models import Document, Patient, TestResult
from app.services.entities import persist_extraction


def _doc(db):
    p = Patient(name="DateTest Person")
    db.add(p); db.flush()
    d = Document(patient_id=p.id, doc_type="lab")
    db.add(d); db.commit(); db.refresh(d)
    return d


def test_persist_sets_report_date_and_observed_at(db):
    d = _doc(db)
    res = ExtractionResult(doc_date="05/10/2023",
                           tests=[ExtractedTest(name="HbA1c", value="6.8", unit="%")])
    persist_extraction(db, document_id=d.id, result=res)
    db.refresh(d)
    assert d.report_date == dt.date(2023, 10, 5)
    tr = db.query(TestResult).order_by(TestResult.id.desc()).first()
    assert tr.observed_at is not None
    assert tr.observed_at.date() == dt.date(2023, 10, 5)


def test_persist_no_date_leaves_null(db):
    d = _doc(db)
    res = ExtractionResult(tests=[ExtractedTest(name="WBC", value="5")])
    persist_extraction(db, document_id=d.id, result=res)
    db.refresh(d)
    assert d.report_date is None
