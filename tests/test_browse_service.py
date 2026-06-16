from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.entities import persist_extraction
from app.services.browse import (
    list_entity_links, list_test_results, list_documents_timeline,
)
from app.agent.state import ExtractionResult, ExtractedEntity, ExtractedTest


def _seed(db):
    p = create_patient(db, name="Browse Pt")
    doc = create_document(db, patient_id=p.id, doc_type="lab_report")
    er = ExtractionResult(
        patient_name="Browse Pt", doc_type="lab_report",
        diseases=[ExtractedEntity(name="anemia", confidence=0.88, source_span="Dx anemia")],
        medications=[ExtractedEntity(name="iron", confidence=0.8, source_span="Rx iron")],
        tests=[ExtractedTest(name="hemoglobin", value="9.1", unit="g/dL",
                             reference_range="12-16", confidence=0.9, source_span="Hb 9.1")],
    )
    persist_extraction(db, document_id=doc.id, result=er)
    return p, doc


def test_list_diseases_with_provenance(db):
    p, doc = _seed(db)
    rows = list_entity_links(db, "disease", patient_id=p.id)
    assert any(r["name"] == "anemia" and r["patient"] == "Browse Pt"
               and r["document_id"] == doc.id and r["source"] == "Dx anemia"
               for r in rows)


def test_list_medications(db):
    p, _ = _seed(db)
    rows = list_entity_links(db, "medication", patient_id=p.id)
    assert any(r["name"] == "iron" for r in rows)


def test_list_test_results(db):
    p, _ = _seed(db)
    rows = list_test_results(db, patient_id=p.id)
    hit = next(r for r in rows if r["test"] == "hemoglobin")
    assert hit["value"] == "9.1"
    assert hit["unit"] == "g/dL"
    assert hit["reference_range"] == "12-16"


def test_documents_timeline(db):
    p, doc = _seed(db)
    rows = list_documents_timeline(db, patient_id=p.id)
    assert any(r["id"] == doc.id and r["type"] == "lab_report" for r in rows)


def test_patient_scope_excludes_others(db):
    p, _ = _seed(db)
    other = create_patient(db, name="Other Browse Pt")
    rows = list_entity_links(db, "disease", patient_id=other.id)
    assert rows == []
