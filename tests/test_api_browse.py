from fastapi.testclient import TestClient

from app.api.server import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "db" in body and "pgvector" in body and "version" in body


def test_create_and_list_patient():
    r = client.post("/api/patients", json={"name": "Api Tester", "age": 40,
                                           "gender": "female"})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["name"] == "Api Tester"
    assert created["image"].startswith("https://i.pravatar.cc/")
    assert created["bloodType"] == "—"

    r2 = client.get("/api/patients")
    assert r2.status_code == 200
    names = [p["name"] for p in r2.json()]
    assert "Api Tester" in names


def test_records_and_documents_empty_for_new_patient():
    pid = client.post("/api/patients", json={"name": "Empty One"}).json()["id"]
    assert client.get(f"/api/patients/{pid}/records").json() == []
    assert client.get(f"/api/patients/{pid}/documents").json() == []


import datetime as dt

from app.db import SessionLocal
from app.models import (
    Document, DocumentEntity, MedicalTest, Patient, TestResult,
)
from app.services import browse as bsvc


def test_list_test_results_prefers_report_date():
    db = SessionLocal()
    try:
        p = Patient(name="Browse Date Person")
        db.add(p); db.flush()
        doc = Document(patient_id=p.id, doc_type="lab",
                       report_date=dt.date(2023, 10, 5))
        db.add(doc); db.flush()
        mt = MedicalTest(name="ESR")
        db.add(mt); db.flush()
        tr = TestResult(medical_test_id=mt.id, value="52", unit="mm/1hr",
                        reference_range="0-15")
        db.add(tr); db.flush()
        db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                              entity_id=tr.id, source_span="ESR 52"))
        db.commit()
        rows = bsvc.list_test_results(db, patient_id=p.id)
        assert rows[0]["date"] == "2023-10-05"  # report_date, not today
        assert rows[0]["reference_range"] == "0-15"
    finally:
        db.close()


def test_delete_records_endpoint():
    from app.db import SessionLocal
    from app.models import Document, Patient
    db = SessionLocal()
    try:
        p = Patient(name="Del Endpoint Person")
        db.add(p); db.flush()
        doc = Document(patient_id=p.id, doc_type="lab")
        db.add(doc); db.commit()
        pid, did = p.id, doc.id
    finally:
        db.close()
    r = client.post(f"/api/patients/{pid}/records/delete",
                    json={"document_ids": [str(did)]})
    assert r.status_code == 200, r.text
    assert r.json()["deleted"] == 1


def test_get_document_file_404_when_missing():
    r = client.get("/api/documents/999999/file")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_get_document_file_serves_existing(tmp_path):
    db = SessionLocal()
    try:
        p = Patient(name="File Owner")
        db.add(p); db.commit(); db.refresh(p)
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        doc = Document(patient_id=p.id, doc_type="LAB REPORT", mime_type="application/pdf",
                       file_path=str(f), original_name="report.pdf")
        db.add(doc); db.commit(); db.refresh(doc)
        did = doc.id
    finally:
        db.close()
    r = client.get(f"/api/documents/{did}/file")
    assert r.status_code == 200
    assert r.content == b"%PDF-1.4 fake"


def test_trends_endpoints():
    import datetime as dt
    from app.db import SessionLocal
    from app.models import Document, DocumentEntity, MedicalTest, Patient, TestResult

    pid = int(client.post("/api/patients", json={"name": "Trend Api"}).json()["id"])
    db = SessionLocal()
    try:
        for d, val in ((dt.date(2024, 1, 1), "14"), (dt.date(2025, 1, 1), "11")):
            doc = Document(patient_id=pid, doc_type="lab report", report_date=d)
            db.add(doc); db.flush()
            mt = MedicalTest(name="Hemoglobin"); db.add(mt); db.flush()
            tr = TestResult(medical_test_id=mt.id, value=val, unit="g/dL",
                            reference_range="12-16")
            db.add(tr); db.flush()
            db.add(DocumentEntity(document_id=doc.id, entity_type="test_result",
                                  entity_id=tr.id))
        db.commit()
    finally:
        db.close()

    metrics = client.get(f"/api/patients/{pid}/trends").json()
    assert any(m["key"] == "hemoglobin" and m["n"] == 2 for m in metrics)

    series = client.get(f"/api/patients/{pid}/trends/hemoglobin").json()
    assert series["ref_low"] == 12.0
    assert len(series["points"]) == 2
    assert series["points"][0]["date"] == "2024-01-01"


def test_trends_unknown_patient_404():
    assert client.get("/api/patients/999999/trends").status_code == 404
