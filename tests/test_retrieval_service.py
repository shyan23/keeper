from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import chunk_and_embed
from app.services.retrieval import search_chunks


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.2] * 768

    def embed_documents(self, texts):
        return [[0.2] * 768 for _ in texts]


def test_search_is_patient_scoped(db):
    pa = create_patient(db, name="Alice Scope")
    pb = create_patient(db, name="Bob Scope")
    da = create_document(db, patient_id=pa.id, doc_type="lab_report")
    dbb = create_document(db, patient_id=pb.id, doc_type="lab_report")
    emb = _FakeEmbedder()
    chunk_and_embed(db, document_id=da.id, patient_id=pa.id,
                    text="alice hemoglobin 13", header="Alice", embedder=emb, size=50)
    chunk_and_embed(db, document_id=dbb.id, patient_id=pb.id,
                    text="bob hemoglobin 14", header="Bob", embedder=emb, size=50)

    hits = search_chunks(db, patient_id=pa.id, query="hemoglobin", embedder=emb, k=5)
    assert hits, "expected at least one hit"
    assert all(h["patient_id"] == pa.id for h in hits)
    assert all("chunk_id" in h and "text" in h for h in hits)
