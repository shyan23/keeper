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
