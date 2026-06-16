import pytest
from pydantic import BaseModel
from app.agent.providers import FallbackChat, FallbackVision, build_embedder


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
