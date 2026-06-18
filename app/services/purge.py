from __future__ import annotations

import os

from sqlalchemy.orm import Session

from app.models import Document, DocumentEntity, TestResult


def delete_documents(db: Session, patient_id: int, document_ids: list[str]) -> int:
    """Delete the given documents (only those owned by patient_id) plus the
    TestResult rows they reference. DocumentEntity and Chunk cascade via FK.
    Shared name tables (Disease/Symptom/Medication/MedicalTest) are left intact.
    Returns the number of documents deleted."""
    ids = []
    for raw in document_ids:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    if not ids:
        return 0
    docs = (db.query(Document)
            .filter(Document.id.in_(ids), Document.patient_id == patient_id)
            .all())
    # A multi-report PDF splits into several documents that share one file hash.
    # Deleting one would otherwise leave its siblings blocking re-upload of the
    # file, so clear the dedup hash on every surviving doc from the same file.
    hashes = {d.content_hash for d in docs if d.content_hash}
    if hashes:
        (db.query(Document)
         .filter(Document.content_hash.in_(hashes),
                 Document.id.notin_(ids))
         .update({Document.content_hash: None}, synchronize_session=False))
    deleted = 0
    for doc in docs:
        # TestResult has no FK to Document — delete via the test_result links first.
        links = (db.query(DocumentEntity)
                 .filter(DocumentEntity.document_id == doc.id,
                         DocumentEntity.entity_type == "test_result")
                 .all())
        tr_ids = [l.entity_id for l in links]
        if tr_ids:
            (db.query(TestResult)
             .filter(TestResult.id.in_(tr_ids))
             .delete(synchronize_session=False))
        if doc.file_path and os.path.exists(doc.file_path):
            try:
                os.remove(doc.file_path)
            except OSError:
                pass
        db.delete(doc)  # cascades DocumentEntity + Chunk
        deleted += 1
    db.commit()
    return deleted
