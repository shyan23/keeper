from __future__ import annotations

from app.cache import get_or_set, make_key
from app.config import get_settings


class OllamaEmbedder:
    """Wraps langchain_ollama.OllamaEmbeddings; `inner` injectable for tests."""

    def __init__(self, inner=None):
        if inner is None:
            from langchain_ollama import OllamaEmbeddings
            s = get_settings()
            inner = OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_host)
            self._model = s.ollama_embed_model
        else:
            self._model = getattr(inner, "model", "inner")
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        # Key includes the model: different models ⇒ different vector spaces.
        key = make_key(f"emb:{self._model}", text)
        return get_or_set(key, lambda: self._inner.embed_query(text))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Per-text caching so a partially-changed corpus reuses unchanged chunks.
        return [self.embed_query(t) for t in texts]
