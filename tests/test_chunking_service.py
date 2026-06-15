from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import make_chunks, chunk_and_embed
from app.models import Chunk


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.2] * 768 for _ in texts]


def test_make_chunks_prefixes_header():
    chunks = make_chunks("line one. line two.", header="Jane · prescription · 2026-06-10", size=12, overlap=0)
    assert len(chunks) >= 2
    assert all(c.startswith("[Jane · prescription · 2026-06-10]") for c in chunks)


def test_chunk_and_embed_persists(db):
    p = create_patient(db, name="Chunk Test")
    doc = create_document(db, patient_id=p.id, doc_type="lab_report")
    n = chunk_and_embed(
        db, document_id=doc.id, patient_id=p.id,
        text="hemoglobin 13.5 normal. wbc 6000 normal.",
        header="Chunk Test · lab_report · 2026-06-10",
        embedder=_FakeEmbedder(), size=20, overlap=0,
    )
    rows = db.query(Chunk).filter_by(document_id=doc.id).all()
    assert n == len(rows) >= 2
    assert rows[0].patient_id == p.id
    assert len(rows[0].embedding) == 768
