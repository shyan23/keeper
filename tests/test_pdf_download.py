import fitz
from fastapi.testclient import TestClient

from app import storage
from app.api.server import app


def test_save_report_and_download():
    client = TestClient(app)
    d = fitz.open(); d.new_page()
    path = storage.save_report(d.tobytes())
    assert path.endswith(".pdf")
    name = path.rsplit("/", 1)[-1]
    r = client.get(f"/api/chat/report/{name}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_download_rejects_path_traversal():
    client = TestClient(app)
    r = client.get("/api/chat/report/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)


def test_download_missing_is_404():
    client = TestClient(app)
    r = client.get("/api/chat/report/deadbeef.pdf")
    assert r.status_code == 404
