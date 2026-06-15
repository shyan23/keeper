from app.services import documents as dsvc
from app.services import patients as psvc


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
