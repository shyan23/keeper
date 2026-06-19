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
        blood_type = None
    out = mapping.patient_to_out(P(), last_visit="2026-06-10")
    assert out["id"] == "7"
    assert out["bloodType"] == "—"  # falls back when unknown
    P.blood_type = "O+"
    assert mapping.patient_to_out(P(), last_visit="")["bloodType"] == "O+"
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
    assert tr["title"] == "HbA1c"
    assert tr["value"] == "6.8"
    assert tr["unit"] == "%"
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


def test_format_value_strips_glued_reference():
    val, ref = mapping.format_value("52  0-15", "mm/1hr", "")
    assert val == "52"
    assert ref == "0-15"


def test_format_value_keeps_existing_reference():
    val, ref = mapping.format_value("6.8 (4.0-6.0)", "%", "4.0-6.0")
    assert val == "6.8"
    assert ref == "4.0-6.0"  # provided ref wins; not overwritten


def test_format_value_normalizes_number():
    val, _ = mapping.format_value(".01", "", "")
    assert val == "0.01"


def test_format_value_passthrough_text():
    val, ref = mapping.format_value("Negative", "", "")
    assert val == "Negative"
    assert ref == ""


def test_merge_records_emits_unit_and_reference():
    tests = [{"test": "ESR", "value": "52  0-15", "unit": "mm/1hr",
              "reference_range": "", "source": "ESR 52", "doc_type": "lab",
              "date": "2023-10-05", "document_id": 3}]
    rows = mapping.merge_records("1", [], [], [], tests)
    tr = rows[0]
    assert tr["title"] == "ESR"
    assert tr["value"] == "52"
    assert tr["unit"] == "mm/1hr"
    assert tr["reference"] == "0-15"      # peeled off the glued value
    assert tr["date"] == "2023-10-05"


def test_merge_records_reference_range_wins():
    tests = [{"test": "HbA1c", "value": "6.8", "unit": "%",
              "reference_range": "4.0-6.0", "document_id": 4}]
    tr = mapping.merge_records("1", [], [], [], tests)[0]
    assert tr["value"] == "6.8"
    assert tr["reference"] == "4.0-6.0"


def test_document_to_out_uses_report_date_and_name():
    from app.api.mapping import document_to_out
    row = {"id": 5, "type": "LAB REPORT", "report_date": "2021-04-30",
           "date": "2026-06-17 08:51", "original_name": "Haematology.pdf", "file": "/x/5.pdf"}
    out = document_to_out(row, "0.2 MB")
    assert out["name"] == "Haematology.pdf"
    assert out["date"] == "2021-04-30"


def test_document_to_out_includes_category():
    from app.api.mapping import document_to_out
    row = {"id": 7, "original_name": "cbc.pdf", "type": "lab report",
           "report_date": "2025-03-04", "date": None, "classification": "Hematology"}
    out = document_to_out(row, "0.1 MB")
    assert out["category"] == "Hematology"

def test_document_to_out_category_none_when_missing():
    from app.api.mapping import document_to_out
    out = document_to_out({"id": 8, "type": "FILE"}, "—")
    assert out["category"] is None
