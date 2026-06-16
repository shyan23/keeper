from pydantic import BaseModel
from app.agent.llm import GroqChat


class _Schema(BaseModel):
    answer: str


class _R:
    def __init__(self, content):
        self.content = content


class _FakeLC:
    def invoke(self, prompt):
        return _R("hello world")

    def bind(self, **kwargs):
        # structured() forces json_object via bind(); return JSON for the schema.
        class _Bound:
            def invoke(self, prompt):
                return _R('{"answer": "grounded"}')
        return _Bound()


def test_chat_complete_returns_text():
    chat = GroqChat(inner=_FakeLC())
    assert chat.complete("hi") == "hello world"


def test_chat_structured_returns_model():
    chat = GroqChat(inner=_FakeLC())
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema)
    assert out.answer == "grounded"
