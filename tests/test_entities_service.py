from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.entities import persist_extraction
from app.agent.state import ExtractionResult, ExtractedEntity, ExtractedTest
from app.models import Disease, DocumentEntity


def test_persist_creates_entities_and_links(db):
    p = create_patient(db, name="Persist Test")
    doc = create_document(db, patient_id=p.id, doc_type="prescription")
    er = ExtractionResult(
        patient_name="Persist Test", doc_type="prescription",
        diseases=[ExtractedEntity(name="asthma", confidence=0.9, source_span="Dx asthma")],
        medications=[ExtractedEntity(name="salbutamol", confidence=0.8, source_span="Rx salbutamol")],
        tests=[ExtractedTest(name="spirometry", value="80", unit="%")],
    )
    n = persist_extraction(db, document_id=doc.id, result=er)
    assert n >= 3  # disease + medication + test linked
    links = db.query(DocumentEntity).filter_by(document_id=doc.id).all()
    assert any(l.entity_type == "disease" and l.validated for l in links)
    # entity de-duplicated by name
    assert db.query(Disease).filter(Disease.name == "asthma").count() == 1
    persist_extraction(db, document_id=doc.id, result=er)
    assert db.query(Disease).filter(Disease.name == "asthma").count() == 1
