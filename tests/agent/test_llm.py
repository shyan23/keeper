import pytest
from pydantic import BaseModel
from app.agent.llm import GroqChat


class _Schema(BaseModel):
    answer: str


class _R:
    def __init__(self, content):
        self.content = content


class _FakeLC:
    """Fake langchain client. Records the response_format bound by structured()."""
    def __init__(self, payload='{"answer": "grounded"}'):
        self.payload = payload
        self.bound_format = None

    def invoke(self, prompt):
        return _R("hello world")

    def bind(self, **kwargs):
        self.bound_format = kwargs.get("response_format")
        payload = self.payload

        class _Bound:
            def invoke(_self, prompt):
                return _R(payload)
        return _Bound()


def _settings(structured_model):
    class S:
        groq_model = "llama-3.3-70b-versatile"
        groq_structured_model = structured_model
        groq_api_key = "x"
    return S()


def test_chat_complete_uses_chat_client():
    chat = GroqChat(inner=_FakeLC())
    assert chat.complete("hi") == "hello world"


def test_structured_strict_path_uses_json_schema(monkeypatch):
    # llm.py does `from app.config import get_settings`, so patch the name bound
    # in the llm module, not app.config.
    monkeypatch.setattr("app.agent.llm.get_settings",
                        lambda: _settings("openai/gpt-oss-120b"))
    fake = _FakeLC()
    chat = GroqChat(structured_inner=fake)
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema) and out.answer == "grounded"
    assert fake.bound_format["type"] == "json_schema"
    assert fake.bound_format["json_schema"]["strict"] is True
    assert fake.bound_format["json_schema"]["name"] == "_Schema"


def test_structured_nonstrict_path_uses_json_object(monkeypatch):
    monkeypatch.setattr("app.agent.llm.get_settings",
                        lambda: _settings("llama-3.3-70b-versatile"))
    fake = _FakeLC()
    chat = GroqChat(structured_inner=fake)
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema) and out.answer == "grounded"
    assert fake.bound_format["type"] == "json_object"


def test_structured_inner_is_lazy():
    # complete() must not build the structured (gpt-oss) client.
    chat = GroqChat(inner=_FakeLC())
    chat.complete("hi")
    assert chat._structured_inner is None
