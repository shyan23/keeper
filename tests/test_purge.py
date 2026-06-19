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


def test_delete_clears_dedup_hash_on_same_file_siblings():
    # Multi-report PDF -> several docs sharing one content_hash. Deleting one
    # must clear the hash on the survivors so the file can be re-uploaded.
    db = SessionLocal()
    try:
        p = Patient(name="Bundle Person")
        db.add(p); db.flush()
        a = Document(patient_id=p.id, doc_type="lab", content_hash="abc123")
        b = Document(patient_id=p.id, doc_type="lab", content_hash="abc123")
        db.add_all([a, b]); db.commit()
        a_id, b_id = a.id, b.id

        n = delete_documents(db, p.id, [str(a_id)])
        assert n == 1
        assert db.get(Document, a_id) is None
        survivor = db.get(Document, b_id)
        assert survivor is not None
        assert survivor.content_hash is None  # dedup freed -> re-upload accepted
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
