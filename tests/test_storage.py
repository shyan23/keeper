from pathlib import Path

import app.storage as storage
from app.storage import save_bytes, path_for, read_file


def test_path_for(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    assert path_for(1, 2, "png") == str(tmp_path / "1" / "2.png")
    assert path_for(1, 2, ".jpg") == str(tmp_path / "1" / "2.jpg")


def test_save_and_read(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "STORAGE_DIR", str(tmp_path))
    p = save_bytes(patient_id=7, document_id=42, ext="pdf", data=b"%PDF-1.4 hi")
    assert Path(p).exists()
    assert p == str(tmp_path / "7" / "42.pdf")
    assert read_file(p) == b"%PDF-1.4 hi"
