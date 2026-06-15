from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Chunk


def make_chunks(text: str, *, header: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Sliding-window character chunks, each prefixed with a contextual header."""
    text = " ".join(text.split())
    if not text:
        return []
    out: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        body = text[start:start + size]
        out.append(f"[{header}] {body}")
        start += step
    return out


def chunk_and_embed(db: Session, *, document_id: int, patient_id: int, text: str,
                    header: str, embedder, size: int = 800, overlap: int = 100) -> int:
    """Chunk text, embed each chunk, persist Chunk rows (with denormalized patient_id). Returns count."""
    chunks = make_chunks(text, header=header, size=size, overlap=overlap)
    if not chunks:
        return 0
    vectors = embedder.embed_documents(chunks)
    for i, (body, vec) in enumerate(zip(chunks, vectors)):
        db.add(Chunk(document_id=document_id, patient_id=patient_id, ord=i,
                     text=body, embedding=vec))
    db.commit()
    return len(chunks)
