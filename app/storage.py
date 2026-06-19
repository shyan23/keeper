import uuid
from pathlib import Path

from app.config import get_settings

STORAGE_DIR = get_settings().storage_dir


def path_for(patient_id: int, document_id: int, ext: str) -> str:
    ext = ext.lstrip(".")
    # Absolute so a server restart from a different cwd still finds the file.
    return str((Path(STORAGE_DIR) / str(patient_id) / f"{document_id}.{ext}").resolve())


def save_staging(ext: str, data: bytes) -> str:
    """Persist an uploaded file before its patient/document are known.
    The ingest agent reads this, then moves it to the patient-scoped path."""
    ext = ext.lstrip(".") or "bin"
    target = (Path(STORAGE_DIR) / "_staging" / f"{uuid.uuid4().hex}.{ext}").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)


def save_bytes(patient_id: int, document_id: int, ext: str, data: bytes) -> str:
    target = Path(path_for(patient_id, document_id, ext))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)


def read_file(path: str) -> bytes:
    return Path(path).read_bytes()


def save_report(data: bytes) -> str:
    """Persist a generated report PDF under STORAGE_DIR/_reports/<uuid>.pdf."""
    target = (Path(STORAGE_DIR) / "_reports" / f"{uuid.uuid4().hex}.pdf").resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return str(target)
