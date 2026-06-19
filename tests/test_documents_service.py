import datetime as dt

from app.services import documents as dsvc
from app.services import patients as psvc


def test_create_document_stores_report_date_and_name(db):
    p = psvc.create_patient(db, name="Date Owner")
    d = dsvc.create_document(db, patient_id=p.id, doc_type="LAB REPORT",
                             report_date=dt.date(2021, 4, 30), original_name="lab.pdf")
    assert d.report_date == dt.date(2021, 4, 30)
    assert d.original_name == "lab.pdf"


def test_create_and_list_documents(db):
    p = psvc.create_patient(db, name="Doc Owner")
    d = dsvc.create_document(db, patient_id=p.id, doc_type="prescription",
                             mime_type="image/png", source_type="image")
    assert d.id is not None
    assert d.patient_id == p.id
    assert d.status == "uploaded"
    assert d.file_path is None
    docs = dsvc.list_documents(db, patient_id=p.id)
    assert [x.id for x in docs] == [d.id]


def test_set_file_path(db):
    p = psvc.create_patient(db, name="Owner2")
    d = dsvc.create_document(db, patient_id=p.id)
    updated = dsvc.set_file_path(db, d.id, "data/files/1/1.png")
    assert updated.file_path == "data/files/1/1.png"


def test_list_all_and_counts(db):
    p = psvc.create_patient(db, name="Owner3")
    dsvc.create_document(db, patient_id=p.id)
    dsvc.create_document(db, patient_id=p.id)
    assert dsvc.count_documents(db) >= 2
    assert dsvc.count_documents(db, patient_id=p.id) == 2


def test_create_document_persists_classification():
    from app.db import SessionLocal
    from app.models import Patient
    from app.services.documents import create_document
    db = SessionLocal()
    try:
        p = Patient(name="Classify Person")
        db.add(p); db.flush()
        doc = create_document(db, patient_id=p.id, doc_type="lab report",
                              classification="Hematology")
        assert doc.classification == "Hematology"
    finally:
        db.close()
