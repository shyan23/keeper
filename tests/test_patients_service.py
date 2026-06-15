import pytest

from app.services import patients as svc


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
