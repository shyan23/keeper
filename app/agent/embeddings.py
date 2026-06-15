from __future__ import annotations

from app.config import get_settings


class OllamaEmbedder:
    """Wraps langchain_ollama.OllamaEmbeddings; `inner` injectable for tests."""

    def __init__(self, inner=None):
        if inner is None:
            from langchain_ollama import OllamaEmbeddings
            s = get_settings()
            inner = OllamaEmbeddings(model=s.ollama_embed_model, base_url=s.ollama_host)
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)
