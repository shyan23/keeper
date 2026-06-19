from app.services.patients import create_patient
from app.services.documents import create_document
from app.services.chunking import chunk_and_embed
from app.services.retrieval import search_chunks


class _FakeEmbedder:
    def embed_query(self, text):
        return [0.2] * 768

    def embed_documents(self, texts):
        return [[0.2] * 768 for _ in texts]


def test_hybrid_keyword_match_ranks_first(db):
    # Constant fake embeddings => vector ranking is a tie; the FTS/BM25 half must
    # break it toward the chunk that lexically contains the query term. Proves the
    # keyword leg of hybrid retrieval + RRF fusion, not just vector recall.
    p = create_patient(db, name="Hybrid Owner")
    d = create_document(db, patient_id=p.id, doc_type="lab_report")
    emb = _FakeEmbedder()
    chunk_and_embed(db, document_id=d.id, patient_id=p.id,
                    text="creatinine level 1.1 mg/dL", header="H", embedder=emb, size=60)
    chunk_and_embed(db, document_id=d.id, patient_id=p.id,
                    text="hemoglobin 13 g/dL platelets normal", header="H", embedder=emb, size=60)
    hits = search_chunks(db, patient_id=p.id, query="creatinine", embedder=emb, k=2)
    assert "creatinine" in hits[0]["text"], hits


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


def test_search_includes_doc_name_and_report_date(db):
    import datetime as dt
    p = create_patient(db, name="Cite Owner")
    d = create_document(db, patient_id=p.id, doc_type="LAB REPORT",
                        report_date=dt.date(2021, 4, 30), original_name="lab.pdf")
    emb = _FakeEmbedder()
    chunk_and_embed(db, document_id=d.id, patient_id=p.id,
                    text="eosinophil 8%", header="Cite", embedder=emb, size=50)
    hits = search_chunks(db, patient_id=p.id, query="eosinophil", embedder=emb, k=5)
    assert hits[0]["report_date"] == "2021-04-30"
    assert hits[0]["original_name"] == "lab.pdf"
    assert "page_ref" in hits[0]
