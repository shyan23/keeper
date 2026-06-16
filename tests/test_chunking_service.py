from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import make_chunks, chunk_and_embed
from app.models import Chunk
from app.services.chunking import make_semantic_chunks


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


class _SemFakeEmbedder:
    def embed_query(self, t):
        return self._v(t)

    def embed_documents(self, ts):
        return [self._v(t) for t in ts]

    def _v(self, t):
        base = [1.0, 0.0] if "alpha" in t else [0.0, 1.0]
        return base + [0.0] * 766


def test_semantic_chunks_splits_on_topic_shift():
    chunks = make_semantic_chunks("alpha one. alpha two. beta three.",
                                  _SemFakeEmbedder(), header="H")
    assert len(chunks) == 2
    assert all(c.startswith("[H]") for c in chunks)
    assert "alpha one. alpha two." in chunks[0]
    assert "beta three." in chunks[1]


def test_semantic_chunks_single_sentence_fallback():
    chunks = make_semantic_chunks("just one sentence", _SemFakeEmbedder(), header="H")
    assert len(chunks) >= 1
    assert chunks[0].startswith("[H]")


def test_chunk_and_embed_accepts_premade_chunks(db):
    from app.services.patients import create_patient
    from app.services.documents import create_document
    from app.services.chunking import chunk_and_embed
    from app.models import Chunk
    p = create_patient(db, name="Premade Chunk Pt")
    doc = create_document(db, patient_id=p.id, doc_type="lab_report")

    class _E:
        def embed_query(self, t):
            return [0.1] * 768
        def embed_documents(self, ts):
            return [[0.2] * 768 for _ in ts]

    n = chunk_and_embed(db, document_id=doc.id, patient_id=p.id, text="ignored",
                        header="H", embedder=_E(), chunks=["[H] a", "[H] b", "[H] c"])
    assert n == 3
    assert db.query(Chunk).filter_by(document_id=doc.id).count() == 3
