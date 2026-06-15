from app.agent.embeddings import OllamaEmbedder


class _FakeInner:
    def embed_query(self, text):
        return [0.1] * 768

    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_embedder_delegates_and_dims():
    emb = OllamaEmbedder(inner=_FakeInner())
    assert len(emb.embed_query("hi")) == 768
    assert len(emb.embed_documents(["a", "b"])) == 2
