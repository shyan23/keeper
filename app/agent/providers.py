from __future__ import annotations
import logging
from typing import Any
from PIL import Image, ImageFilter, ImageOps
from pydantic import BaseModel
import io
from app.agent.embeddings import OllamaEmbedder
from app.agent.llm import GroqChat, GroqVision
from app.config import get_settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
import pytesseract
import base64
log = logging.getLogger(__name__)


# ---- Gemini wrappers (lazy import so the module loads without the dep) ----

class GeminiChat:
    def __init__(self, inner=None):
        if inner is None:
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

    Phone scans (CamScanner etc.) OCR badly in a single default pass. So ocr_image
    runs an escalating set of preprocessing variants and keeps the result with the
    highest mean per-word confidence, stopping early once a variant clears the
    benchmark. Tesseract-only — no vision-model escalation (CPU/speed budget)."""

    # Tesseract confidence is 0–100. Stop as soon as a variant's mean word
    # confidence clears this; otherwise return the best variant tried.
    BENCHMARK = 70.0

    def ocr_image(self, data: bytes, mime: str) -> str:
        return self.ocr_with_confidence(data, mime)[0]

    def ocr_with_confidence(self, data: bytes, mime: str) -> tuple[str, float]:
        """Like ocr_image but also returns the best variant's mean word confidence
        (0-100), so callers can decide whether to escalate (see PrescriptionVision)."""
        variants = self._preprocess_variants(data, mime)
        best_text, best_conf, best_name = "", -1.0, "none"
        for name, img, psm in variants:
            try:
                text, conf = self._ocr_pass(img, psm)
            except Exception as e:  # noqa: BLE001 - one bad variant must not abort the loop
                log.warning("ocr variant %s failed: %s", name, e)
                continue
            log.info("ocr variant %s: conf=%.1f len=%d", name, conf, len(text))
            if conf > best_conf:
                best_text, best_conf, best_name = text, conf, name
            if conf >= self.BENCHMARK:
                break
        log.info("ocr best variant=%s conf=%.1f (benchmark %.0f)",
                 best_name, best_conf, self.BENCHMARK)
        return best_text.strip(), max(best_conf, 0.0)

    def _preprocess_variants(self, data: bytes, mime: str):
        """Yield (name, PIL.Image, psm) tuples, cheapest/most-likely first so the
        loop usually stops on variant 1. All PIL-only — no numpy/cv2."""
        base = Image.open(io.BytesIO(data))
        gray = ImageOps.grayscale(base)
        # Upscale small scans: Tesseract wants ~300 DPI; phone shots are often less.
        scale = 2 if max(gray.size) < 2000 else 1
        big = gray.resize((gray.width * scale, gray.height * scale)) if scale > 1 else gray
        contrast = ImageOps.autocontrast(big)
        sharp = contrast.filter(ImageFilter.SHARPEN)
        binar = sharp.point(lambda p: 255 if p > 150 else 0, mode="L")  # crude global threshold
        return [
            ("raw", base, 3),               # default: clean printed docs win here
            ("gray-up-contrast", contrast, 6),  # assume a single uniform block
            ("sharpen", sharp, 4),          # assume a column of text
            ("binarize", binar, 6),         # last resort for low-contrast scans
        ]

    def _ocr_pass(self, img, psm: int) -> tuple[str, float]:
        """Run one Tesseract pass; return (text, mean_word_confidence 0–100)."""
        cfg = f"--oem 3 --psm {psm}"
        data = pytesseract.image_to_data(img, config=cfg,
                                         output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for word, conf in zip(data["text"], data["conf"]):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if c >= 0 and word.strip():
                words.append(word)
                confs.append(c)
        text = " ".join(words)
        mean = sum(confs) / len(confs) if confs else 0.0
        return text, mean


class OllamaVision:
    """Local OCR fallback via an Ollama vision model (e.g. llama3.2-vision).
    Requires `ollama pull <ollama_vision_model>`; only hit if cloud vision fails."""

    def __init__(self, inner=None):
        if inner is None:
            s = get_settings()
            inner = ChatOllama(model=s.ollama_vision_model, base_url=s.ollama_host, temperature=0)
        self._inner = inner

    def ocr_image(self, data: bytes, mime: str) -> str:
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


class PrescriptionVision:
    """OCR via Tesseract; escalate to Gemini ONLY when the page looks handwritten.

    Tesseract handles every printed document (free, CPU). When its output looks
    like a handwritten prescription (low confidence, or near-empty for Bangla),
    re-OCR that page with the paid Gemini vision model — the sole place Gemini is
    used. Gemini failure falls back to the Tesseract text (never blocks ingestion).
    The OCR cache (extraction._extract_raw) means each page is paid for at most once."""

    def __init__(self, tesseract: TesseractVision, gemini=None):
        self._tess = tesseract
        self._gemini = gemini

    def ocr_image(self, data: bytes, mime: str) -> str:
        from app.services.handwriting import looks_like_handwriting
        text, conf = self._tess.ocr_with_confidence(data, mime)
        if self._gemini is not None and looks_like_handwriting(text, conf):
            log.info("handwriting detected (conf=%.1f, len=%d) -> Gemini OCR",
                     conf, len(text))
            try:
                return self._gemini.ocr_image(data, mime)
            except Exception as e:  # noqa: BLE001 - paid path is best-effort
                log.warning("Gemini prescription OCR failed, keeping Tesseract: %s", e)
        return text


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

    ai_provider="groq" (default): Groq primary, Ollama fallback.
    ai_provider="ollama": local-only, no cloud calls.
    ai_provider="fallback": Gemini->Groq->Ollama chat chain (cloud first).
    OCR: Tesseract (CPU) for printed docs; handwritten prescriptions escalate to
    the paid Gemini vision model — see PrescriptionVision.
    """
    s = get_settings()
    chats = []
    if s.ai_provider == "fallback":
        if _has_gemini():
            chats.append(GeminiChat())
        if s.groq_api_key:
            chats.append(GroqChat())
    elif s.ai_provider == "groq" and s.groq_api_key:
        chats.append(GroqChat())
    chats.append(OllamaChat())  # always last: offline fallback

    # OCR: Tesseract for printed docs; escalate handwritten prescriptions to the
    # paid Gemini vision model (only when Gemini is configured).
    gemini_vision = GeminiVision() if _has_gemini() else None
    visions = [PrescriptionVision(TesseractVision(), gemini_vision)]

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
