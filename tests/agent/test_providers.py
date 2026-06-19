import pytest
from pydantic import BaseModel
from app.agent.providers import FallbackChat, FallbackVision, TesseractVision, build_embedder


class _Schema(BaseModel):
    answer: str


class _BoomChat:
    def complete(self, prompt):
        raise RuntimeError("provider down")
    def structured(self, prompt, schema):
        raise RuntimeError("provider down")


class _OkChat:
    def complete(self, prompt):
        return "ok-text"
    def structured(self, prompt, schema):
        return schema(answer="ok-struct")


class _BoomVision:
    def ocr_image(self, data, mime):
        raise RuntimeError("vision down")


class _OkVision:
    def ocr_image(self, data, mime):
        return "ocr-ok"


class _BoomEmbedder:
    def embed_query(self, text):
        raise RuntimeError("embed down")
    def embed_documents(self, texts):
        raise RuntimeError("embed down")


class _OkEmbedder:
    def embed_query(self, text):
        return [0.1] * 768
    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]


def test_fallback_chat_advances_past_failure():
    chat = FallbackChat([_BoomChat(), _OkChat()])
    assert chat.complete("hi") == "ok-text"
    assert chat.structured("x", _Schema).answer == "ok-struct"


def test_fallback_chat_all_fail_raises():
    chat = FallbackChat([_BoomChat(), _BoomChat()])
    with pytest.raises(RuntimeError):
        chat.complete("hi")


def test_fallback_vision_advances():
    v = FallbackVision([_BoomVision(), _OkVision()])
    assert v.ocr_image(b"x", "image/png") == "ocr-ok"


def test_build_embedder_picks_first_working():
    emb = build_embedder([_BoomEmbedder(), _OkEmbedder()])
    assert len(emb.embed_query("ping")) == 768


def test_build_embedder_none_working_raises():
    with pytest.raises(RuntimeError):
        build_embedder([_BoomEmbedder()])


def test_fallback_skips_none_providers():
    chat = FallbackChat([None, _OkChat()])
    assert chat.complete("hi") == "ok-text"


def _stub_variants(monkeypatch):
    """Replace preprocessing with 3 dummy named passes (image is irrelevant)."""
    variants = [("a", None, 3), ("b", None, 6), ("c", None, 4)]
    monkeypatch.setattr(TesseractVision, "_preprocess_variants",
                        lambda self, data, mime: variants)


def test_ocr_loop_stops_at_benchmark(monkeypatch):
    """Loop returns the first variant clearing BENCHMARK and skips the rest."""
    _stub_variants(monkeypatch)
    calls = []
    results = {3: ("garbage", 20.0), 6: ("clean report text", 85.0), 4: ("late", 99.0)}

    def fake_pass(self, img, psm):
        calls.append(psm)
        return results[psm]

    monkeypatch.setattr(TesseractVision, "_ocr_pass", fake_pass)
    out = TesseractVision().ocr_image(b"x", "image/png")
    assert out == "clean report text"
    assert calls == [3, 6]  # stopped after benchmark hit; variant 4 never tried


def test_ocr_loop_keeps_best_when_none_reach_benchmark(monkeypatch):
    """When no variant clears BENCHMARK, return the highest-confidence one."""
    _stub_variants(monkeypatch)
    results = {3: ("low", 10.0), 6: ("best below bar", 55.0), 4: ("mid", 40.0)}
    monkeypatch.setattr(TesseractVision, "_ocr_pass",
                        lambda self, img, psm: results[psm])
    out = TesseractVision().ocr_image(b"x", "image/png")
    assert out == "best below bar"


def test_ocr_loop_skips_failing_variant(monkeypatch):
    """A variant that raises is logged and skipped, not fatal."""
    _stub_variants(monkeypatch)

    def fake_pass(self, img, psm):
        if psm == 3:
            raise RuntimeError("tesseract boom")
        return {6: ("recovered", 90.0), 4: ("x", 95.0)}[psm]

    monkeypatch.setattr(TesseractVision, "_ocr_pass", fake_pass)
    out = TesseractVision().ocr_image(b"x", "image/png")
    assert out == "recovered"
