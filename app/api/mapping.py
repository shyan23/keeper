from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

# browse entity_type -> UI record type
_ENTITY_TO_UI = {"disease": "disease", "symptom": "symptom", "medication": "medicine"}


def avatar_url(name: str) -> str:
    return f"https://i.pravatar.cc/150?u={quote(name or '')}"


def patient_to_out(patient, last_visit: str | None) -> dict:
    return {
        "id": str(patient.id),
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "bloodType": "—",
        "image": avatar_url(patient.name),
        "lastVisit": last_visit or "",
        "status": "Active",
    }


def _record(patient_id: str, ui_type: str, idx: int, title: str,
            row: dict) -> dict:
    return {
        "id": f"{ui_type}-{row.get('document_id')}-{idx}",
        "patientId": patient_id,
        "type": ui_type,
        "title": title,
        "description": (row.get("source") or row.get("doc_type") or ""),
        "date": row.get("date"),
        "status": "Recorded",
        "severity": None,
        "doctor": None,
    }


def merge_records(patient_id: str, diseases: list[dict], symptoms: list[dict],
                  medications: list[dict], tests: list[dict]) -> list[dict]:
    out: list[dict] = []
    idx = 0
    for ui_type, rows in (("disease", diseases), ("symptom", symptoms),
                          ("medicine", medications)):
        for r in rows:
            out.append(_record(patient_id, ui_type, idx, r.get("name") or "", r))
            idx += 1
    for r in tests:
        value = "".join(x for x in [r.get("value"), r.get("unit")] if x)
        title = f"{r.get('test') or ''}: {value}".strip().rstrip(":")
        out.append(_record(patient_id, "test_result", idx, title, r))
        idx += 1
    return out


def document_to_out(row: dict, size_str: str) -> dict:
    file_path = row.get("file") or ""
    name = Path(file_path).name if file_path else f"document-{row.get('id')}"
    return {
        "id": str(row.get("id")),
        "name": name,
        "date": row.get("date"),
        "type": (row.get("type") or "FILE"),
        "size": size_str,
    }
