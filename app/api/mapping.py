from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import quote

# browse entity_type -> UI record type
_ENTITY_TO_UI = {"disease": "disease", "symptom": "symptom", "medication": "medicine"}

_REF_RE = re.compile(r"\(?\s*(\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?)\s*\)?")


def format_value(value: str | None, unit: str | None, reference: str | None) -> tuple[str, str]:
    """Return (clean_value, reference). Split a glued 'value  low-high' interval off the
    value; normalize numbers; pass non-numeric values through unchanged."""
    ref = (reference or "").strip()
    v = (value or "").strip()
    if not v:
        return "", ref
    # If a low-high interval is embedded in the value, peel it off.
    m = _REF_RE.search(v)
    if m and re.match(r"^\s*\d", v):
        interval = m.group(1).replace(" ", "")
        head = v[:m.start()].strip()
        if head:
            v = head
            if not ref:
                ref = interval
    # Normalize a leading bare number (".01" -> "0.01", "6.80" -> "6.8").
    nm = re.match(r"^-?\d*\.?\d+", v)
    if nm and nm.group(0) == v:
        try:
            f = float(v)
            v = ("%g" % f)
        except ValueError:
            pass
    return v, ref


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


def _record(patient_id: str, ui_type: str, idx: int, title: str, row: dict,
            value: str = "", unit: str = "", reference: str = "") -> dict:
    return {
        "id": f"{ui_type}-{row.get('document_id')}-{idx}",
        "patientId": patient_id,
        "type": ui_type,
        "title": title,
        "description": (row.get("source") or row.get("doc_type") or ""),
        "value": value,
        "unit": unit,
        "reference": reference,
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
        value, reference = format_value(r.get("value"), r.get("unit"), r.get("reference_range"))
        out.append(_record(patient_id, "test_result", idx, r.get("test") or "", r,
                           value=value, unit=(r.get("unit") or ""), reference=reference))
        idx += 1
    return out


def document_to_out(row: dict, size_str: str) -> dict:
    file_path = row.get("file") or ""
    name = (row.get("original_name")
            or (Path(file_path).name if file_path else None)
            or f"document-{row.get('id')}")
    # Prefer the date printed on the report; fall back to upload time.
    date = row.get("report_date") or row.get("date")
    return {
        "id": str(row.get("id")),
        "name": name,
        "date": date,
        "type": (row.get("type") or "FILE"),
        "size": size_str,
    }
