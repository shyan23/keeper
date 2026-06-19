"""MCP server — exposes MedAgentic records + retrieval as tools any MCP client
(Claude Desktop, Continue, …) can drive. Pure local, no egress.

Run:  python -m app.mcp.server      (stdio transport)
   or: mcp run app/mcp/server.py

Read/query only by design. Ingestion runs through a human-in-the-loop patient
gate (see the agent graph) that can't be driven headlessly without losing the
safety check, so it's intentionally not an MCP tool yet — drive ingest from the
dashboard. Add an MCP ingest tool only once the HITL gate has a headless contract.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.api import mapping
from app.db import SessionLocal
from app.services import browse as bsvc
from app.services import patients as psvc

mcp = FastMCP("medagentic")


@mcp.tool()
def list_patients() -> list[dict]:
    """List every patient (family member) with id, name, age, gender, blood type."""
    db = SessionLocal()
    try:
        return [{"id": p.id, "name": p.name, "age": p.age,
                 "gender": p.gender, "blood_type": p.blood_type}
                for p in psvc.list_patients(db)]
    finally:
        db.close()


@mcp.tool()
def get_records(patient_id: int) -> list[dict]:
    """Get a patient's structured medical records: diseases, symptoms, medications,
    and test results, merged into the same shape the dashboard shows."""
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise ValueError(f"patient {patient_id} not found")
        diseases = bsvc.list_entity_links(db, "disease", patient_id=patient_id)
        symptoms = bsvc.list_entity_links(db, "symptom", patient_id=patient_id)
        meds = bsvc.list_entity_links(db, "medication", patient_id=patient_id)
        tests = bsvc.list_test_results(db, patient_id=patient_id)
        return mapping.merge_records(str(patient_id), diseases, symptoms, meds, tests)
    finally:
        db.close()


@mcp.tool()
def search_records(patient_id: int, query: str, k: int = 5) -> list[dict]:
    """Hybrid (semantic + keyword) search over a patient's document chunks. Returns
    the top-k matching passages with citation metadata (document, page, date) so the
    calling agent can compose a source-backed answer."""
    from app.api.runtime import get_deps
    from app.services.retrieval import search_chunks
    db = SessionLocal()
    try:
        if psvc.get_patient(db, patient_id) is None:
            raise ValueError(f"patient {patient_id} not found")
        return search_chunks(db, patient_id=patient_id, query=query,
                             embedder=get_deps().embedder, k=k)
    finally:
        db.close()


if __name__ == "__main__":
    mcp.run()
