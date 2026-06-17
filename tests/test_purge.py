from app.db import SessionLocal
from app.models import (
    Chunk, Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services.purge import delete_documents


def _seed(db):
    p = Patient(name="Purge Person")
    db.add(p); db.flush()
    doc = Document(patient_id=p.id, doc_type="lab")
    db.add(doc); db.flush()
    mt = MedicalTest(name="WBC")
    db.add(mt); db.flush()
    tr = TestResult(medical_test_id=mt.id, value="5")
    db.add(tr); db.flush()
    db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                          entity_id=tr.id))
    db.add(Chunk(document_id=doc.id, patient_id=p.id, ord=0, text="hi"))
    db.commit()
    return p.id, doc.id, tr.id


def test_delete_documents_removes_doc_entities_chunks_testresults():
    db = SessionLocal()
    try:
        pid, doc_id, tr_id = _seed(db)
        n = delete_documents(db, pid, [str(doc_id)])
        assert n == 1
        assert db.get(Document, doc_id) is None
        assert db.get(TestResult, tr_id) is None
        assert db.query(DocumentEntity).filter_by(document_id=doc_id).count() == 0
        assert db.query(Chunk).filter_by(document_id=doc_id).count() == 0
    finally:
        db.close()


def test_delete_documents_skips_foreign_patient():
    db = SessionLocal()
    try:
        pid, doc_id, _ = _seed(db)
        other = Patient(name="Other")
        db.add(other); db.flush()
        db.commit()
        n = delete_documents(db, other.id, [str(doc_id)])  # wrong owner
        assert n == 0
        assert db.get(Document, doc_id) is not None
    finally:
        db.close()
