from __future__ import annotations

import math
import re

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


def _split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _cos_dist(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def make_semantic_chunks(text: str, embedder, *, header: str,
                         max_chars: int = 1200, threshold_pct: float = 90) -> list[str]:
    """Group sentences into chunks, splitting where consecutive-sentence embedding
    distance exceeds the threshold_pct percentile (a topic shift) or max_chars is hit.
    Falls back to fixed-size chunking when there is <=1 sentence."""
    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return make_chunks(text, header=header)
    embs = embedder.embed_documents(sentences)
    dists = [_cos_dist(embs[i], embs[i + 1]) for i in range(len(embs) - 1)]
    threshold = _percentile(dists, threshold_pct)
    groups: list[str] = []
    cur = [sentences[0]]
    for i in range(1, len(sentences)):
        boundary = dists[i - 1] > threshold
        too_long = sum(len(x) + 1 for x in cur) + len(sentences[i]) > max_chars
        if boundary or too_long:
            groups.append(" ".join(cur))
            cur = [sentences[i]]
        else:
            cur.append(sentences[i])
    groups.append(" ".join(cur))
    return [f"[{header}] {g}" for g in groups]


def chunk_and_embed(db: Session, *, document_id: int, patient_id: int, text: str,
                    header: str, embedder, size: int = 800, overlap: int = 100,
                    chunks: list[str] | None = None) -> int:
    """Chunk text (or use pre-made `chunks`), embed each, persist Chunk rows. Returns count."""
    if chunks is None:
        chunks = make_chunks(text, header=header, size=size, overlap=overlap)
    if not chunks:
        return 0
    vectors = embedder.embed_documents(chunks)
    for i, (body, vec) in enumerate(zip(chunks, vectors)):
        db.add(Chunk(document_id=document_id, patient_id=patient_id, ord=i,
                     text=body, embedding=vec))
    db.commit()
    return len(chunks)
