from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document


def search_chunks(db: Session, *, patient_id: int, query: str, embedder, k: int = 5) -> list[dict]:
    """Patient-scoped pgvector cosine search. Returns dicts with proof metadata."""
    qvec = embedder.embed_query(query)
    stmt = (
        select(Chunk, Document.doc_type, Document.uploaded_at)
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.patient_id == patient_id)
        .order_by(Chunk.embedding.cosine_distance(qvec))
        .limit(k)
    )
    rows = db.execute(stmt).all()
    out: list[dict] = []
    for chunk, doc_type, uploaded_at in rows:
        out.append({
            "chunk_id": chunk.id,
            "document_id": chunk.document_id,
            "patient_id": chunk.patient_id,
            "text": chunk.text,
            "doc_type": doc_type,
            "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
        })
    return out
