from app.api.schemas import (
    DocumentOut, HealthOut, PatientIn, PatientOut, RecordOut,
)


def test_patient_out_serializes():
    p = PatientOut(id="1", name="Jane", age=42, gender="female",
                   bloodType="—", image="http://x", lastVisit="2026-06-01",
                   status="Active")
    assert p.model_dump()["id"] == "1"


def test_patient_in_optional_fields():
    p = PatientIn(name="Jane")
    assert p.age is None and p.gender is None and p.relationship is None


from app.api import mapping


def test_avatar_url_encodes_name():
    url = mapping.avatar_url("Jane Doe")
    assert url == "https://i.pravatar.cc/150?u=Jane%20Doe"


def test_patient_to_out_derives_fields():
    class P:  # minimal stand-in for the Patient ORM row
        id = 7
        name = "Jane Doe"
        age = 30
        gender = "female"
    out = mapping.patient_to_out(P(), last_visit="2026-06-10")
    assert out["id"] == "7"
    assert out["bloodType"] == "—"
    assert out["image"] == "https://i.pravatar.cc/150?u=Jane%20Doe"
    assert out["lastVisit"] == "2026-06-10"
    assert out["status"] == "Active"


def test_merge_records_maps_types_and_titles():
    diseases = [{"name": "Diabetes", "source": "dx span", "doc_type": "lab",
                 "date": "2026-05-01", "document_id": 3}]
    symptoms = [{"name": "Fatigue", "source": None, "doc_type": None,
                 "date": "2026-05-01", "document_id": 3}]
    meds = [{"name": "Metformin", "source": "rx", "doc_type": "rx",
             "date": "2026-05-02", "document_id": 4}]
    tests = [{"test": "HbA1c", "value": "6.8", "unit": "%", "source": "lab span",
              "doc_type": "lab", "date": "2026-05-01", "document_id": 3}]
    rows = mapping.merge_records("1", diseases, symptoms, meds, tests)
    by_type = {r["type"] for r in rows}
    assert by_type == {"disease", "symptom", "medicine", "test_result"}
    med = next(r for r in rows if r["type"] == "medicine")
    assert med["title"] == "Metformin"
    tr = next(r for r in rows if r["type"] == "test_result")
    assert tr["title"] == "HbA1c: 6.8%"
    assert all(r["patientId"] == "1" for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)  # ids unique


def test_merge_records_empty():
    assert mapping.merge_records("1", [], [], [], []) == []


def test_document_to_out_basename_and_size():
    row = {"id": 9, "file": "/data/files/3/9.pdf", "type": "lab",
           "date": "2026-06-01 10:00"}
    out = mapping.document_to_out(row, size_str="1.2 MB")
    assert out["id"] == "9"
    assert out["name"] == "9.pdf"
    assert out["type"] == "lab"
    assert out["size"] == "1.2 MB"
