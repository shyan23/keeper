from datetime import date

import pytest

from app.services import patients as svc
from app.services.documents import create_document


def test_create_and_get(db):
    p = svc.create_patient(db, name="Asha", age=30, relationship="mother")
    assert p.id is not None
    fetched = svc.get_patient(db, p.id)
    assert fetched.name == "Asha"
    assert fetched.relationship == "mother"


def test_list_orders_by_id(db):
    a = svc.create_patient(db, name="A")
    b = svc.create_patient(db, name="B")
    ids = [p.id for p in svc.list_patients(db)]
    assert ids == sorted(ids)
    assert a.id in ids and b.id in ids


def test_update(db):
    p = svc.create_patient(db, name="Asha", age=30)
    updated = svc.update_patient(db, p.id, age=31)
    assert updated.age == 31


def test_delete(db):
    p = svc.create_patient(db, name="Asha")
    svc.delete_patient(db, p.id)
    assert svc.get_patient(db, p.id) is None


def test_get_missing_returns_none(db):
    assert svc.get_patient(db, 999999) is None


def test_update_missing_raises(db):
    with pytest.raises(ValueError):
        svc.update_patient(db, 999999, age=1)


def test_apply_latest_demographics_tracks_newest_document(db):
    p = svc.create_patient(db, name="Nafisa", age=40)
    # Older report on file (Jan); a Feb report should win for the header age + blood.
    create_document(db, patient_id=p.id, report_date=date(2026, 1, 1))
    create_document(db, patient_id=p.id, report_date=date(2026, 2, 1))
    svc.apply_latest_demographics(db, p.id, age=48, blood_type="A+",
                                  doc_date=date(2026, 2, 1))
    assert svc.get_patient(db, p.id).age == 48
    assert svc.get_patient(db, p.id).blood_type == "A+"

    # An OLDER report uploaded afterwards must not roll the age back…
    svc.apply_latest_demographics(db, p.id, age=39, doc_date=date(2025, 12, 1))
    assert svc.get_patient(db, p.id).age == 48
    # …but a still-blank field is filled regardless of date.
    svc.apply_latest_demographics(db, p.id, gender="female", doc_date=date(2025, 1, 1))
    assert svc.get_patient(db, p.id).gender == "female"
