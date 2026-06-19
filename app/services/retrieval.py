from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Chunk, Document

# Reciprocal-rank-fusion constant. Standard 60: damps the top rank's dominance so
# a chunk strong in only one retriever still ranks above chunks weak in both.
_RRF_K = 60


def _row_to_hit(chunk: Chunk, doc_type, uploaded_at, report_date, original_name) -> dict:
    return {
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "patient_id": chunk.patient_id,
        "text": chunk.text,
        "doc_type": doc_type,
        "report_date": report_date.isoformat() if report_date else None,
        "original_name": original_name,
        "page_ref": chunk.page_ref,
        "uploaded_at": uploaded_at.isoformat() if uploaded_at else None,
    }


def search_chunks(db: Session, *, patient_id: int, query: str, embedder, k: int = 5) -> list[dict]:
    """Patient-scoped HYBRID retrieval: pgvector cosine (semantic) fused with
    Postgres full-text search (BM25-ish keyword) via reciprocal rank fusion.

    Vector recall alone misses exact tokens (a test code, a drug name) when the
    query and chunk are semantically near but lexically different — and vice
    versa. RRF merges both ranked lists without tuning a weight: a chunk's score
    is the sum of 1/(_RRF_K + rank) over the lists it appears in. Returns the top
    k fused hits with the same proof metadata as before.
    """
    cols = (Chunk, Document.doc_type, Document.uploaded_at,
            Document.report_date, Document.original_name)
    base = select(*cols).join(Document, Document.id == Chunk.document_id).where(
        Chunk.patient_id == patient_id)

    # Pull a wider candidate pool from each retriever than k, so fusion has room.
    pool = max(k * 4, 20)

    qvec = embedder.embed_query(query)
    vec_rows = db.execute(
        base.order_by(Chunk.embedding.cosine_distance(qvec)).limit(pool)
    ).all()

    # websearch_to_tsquery tolerates raw user text (quotes, "or", bare words).
    tsq = func.websearch_to_tsquery("english", query)
    tsv = func.to_tsvector("english", Chunk.text)
    fts_rows = db.execute(
        base.where(tsv.op("@@")(tsq))
            .order_by(func.ts_rank(tsv, tsq).desc())
            .limit(pool)
    ).all()

    scores: dict[int, float] = {}
    rows_by_id: dict[int, tuple] = {}
    for ranked in (vec_rows, fts_rows):
        for rank, row in enumerate(ranked):
            cid = row[0].id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
            rows_by_id[cid] = row

    top = sorted(scores, key=scores.get, reverse=True)[:k]
    return [_row_to_hit(*rows_by_id[cid]) for cid in top]
