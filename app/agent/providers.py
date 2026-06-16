from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from app.agent.embeddings import OllamaEmbedder
from app.agent.llm import GroqChat, GroqVision
from app.config import get_settings

log = logging.getLogger(__name__)


# ---- Gemini wrappers (lazy import so the module loads without the dep) ----

class GeminiChat:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            s = get_settings()
            inner = ChatGoogleGenerativeAI(model=s.gemini_model, max_retries=0,
                                           google_api_key=s.gemini_api_key, temperature=0)
        self._inner = inner

    def complete(self, prompt: str) -> str:
        return self._inner.invoke(prompt).content

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self._inner.with_structured_output(schema).invoke(prompt)


class GeminiVision:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            s = get_settings()
            inner = ChatGoogleGenerativeAI(model=s.gemini_vision_model, max_retries=0,
                                           google_api_key=s.gemini_api_key, temperature=0)
        self._inner = inner

    def ocr_image(self, data: bytes, mime: str) -> str:
        import base64
        b64 = base64.b64encode(data).decode()
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe ALL text in this medical document verbatim. Output only the text."},
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            ],
        }]
        return self._inner.invoke(msg).content


class GeminiEmbedder:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            s = get_settings()
            inner = GoogleGenerativeAIEmbeddings(model=s.gemini_embed_model,
                                                 google_api_key=s.gemini_api_key)
        self._inner = inner

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)


class OllamaChat:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_ollama import ChatOllama
            s = get_settings()
            inner = ChatOllama(model=s.ollama_model, base_url=s.ollama_host, temperature=0)
        self._inner = inner

    def complete(self, prompt: str) -> str:
        return self._inner.invoke(prompt).content

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        return self._inner.with_structured_output(schema).invoke(prompt)


class TesseractVision:
    """CPU OCR via Tesseract — no GPU/RAM cost, strong on printed lab reports.

    Needs the system binary (`apt install tesseract-ocr`) + pytesseract. If either
    is missing, ocr_image raises and FallbackVision advances to the next provider.
    """

    def ocr_image(self, data: bytes, mime: str) -> str:
        import io

        import pytesseract
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(img).strip()


class OllamaVision:
    """Local OCR fallback via an Ollama vision model (e.g. llama3.2-vision).
    Requires `ollama pull <ollama_vision_model>`; only hit if cloud vision fails."""

    def __init__(self, inner=None):
        if inner is None:
            from langchain_ollama import ChatOllama
            s = get_settings()
            inner = ChatOllama(model=s.ollama_vision_model, base_url=s.ollama_host, temperature=0)
        self._inner = inner

    def ocr_image(self, data: bytes, mime: str) -> str:
        import base64
        b64 = base64.b64encode(data).decode()
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe ALL text in this medical document verbatim. Output only the text."},
                {"type": "image_url", "image_url": f"data:{mime};base64,{b64}"},
            ],
        }]
        return self._inner.invoke(msg).content


# ---- Fallback wrappers ----

class FallbackChat:
    """Try each provider in order; advance on any exception."""

    def __init__(self, providers: list):
        self._providers = [p for p in providers if p is not None]

    def complete(self, prompt: str) -> str:
        last = None
        for p in self._providers:
            try:
                return p.complete(prompt)
            except Exception as e:  # noqa: BLE001 - fallback is the whole point
                log.warning("chat provider %s failed: %s", type(p).__name__, e)
                last = e
        raise RuntimeError(f"all chat providers failed: {last}")

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        last = None
        for p in self._providers:
            try:
                return p.structured(prompt, schema)
            except Exception as e:  # noqa: BLE001
                log.warning("chat(structured) provider %s failed: %s", type(p).__name__, e)
                last = e
        raise RuntimeError(f"all chat providers failed: {last}")


class FallbackVision:
    def __init__(self, providers: list):
        self._providers = [p for p in providers if p is not None]

    def ocr_image(self, data: bytes, mime: str) -> str:
        last = None
        for p in self._providers:
            try:
                return p.ocr_image(data, mime)
            except Exception as e:  # noqa: BLE001
                log.warning("vision provider %s failed: %s", type(p).__name__, e)
                last = e
        raise RuntimeError(f"all vision providers failed: {last}")


def build_embedder(candidates: list):
    """Sticky embedder: probe each candidate once; return the first that works.
    Never falls back mid-corpus (would mix incompatible vector spaces)."""
    last = None
    for e in candidates:
        if e is None:
            continue
        try:
            e.embed_query("ping")
            return e
        except Exception as exc:  # noqa: BLE001
            log.warning("embedder %s unavailable: %s", type(e).__name__, exc)
            last = exc
    raise RuntimeError(f"no embedder available: {last}")


def _has_gemini() -> bool:
    s = get_settings()
    return bool(s.gemini_api_key) and s.gemini_api_key != "changeme"


def build_deps(session_factory: Any):
    """Construct production Deps.

    ai_provider="ollama" (default): local-only, no cloud calls.
    ai_provider="fallback": Gemini->Groq->Ollama chat chain (cloud first).
    OCR is always Tesseract (CPU) — no vision models.
    """
    s = get_settings()
    chats = []
    if s.ai_provider == "fallback":
        if _has_gemini():
            chats.append(GeminiChat())
        if s.groq_api_key:
            chats.append(GroqChat())
    chats.append(OllamaChat())

    # OCR is Tesseract only — no vision models (CPU, ~0 RAM, deterministic).
    visions = [TesseractVision()]

    # Embeddings are pinned to a single provider (Ollama nomic-embed-text, 768-dim):
    # mixing embedding providers corrupts the shared pgvector space, and current
    # Gemini embedding models don't match the fixed vector(768) column.
    embed_candidates = [OllamaEmbedder()]

    from app.agent.state import Deps
    return Deps(
        chat=FallbackChat(chats),
        vision=FallbackVision(visions),
        embedder=build_embedder(embed_candidates),
        session_factory=session_factory,
    )
