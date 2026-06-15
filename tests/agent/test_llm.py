from pydantic import BaseModel
from app.agent.llm import GroqChat


class _Schema(BaseModel):
    answer: str


class _FakeLC:
    def invoke(self, prompt):
        class _R:
            content = "hello world"
        return _R()

    def with_structured_output(self, schema):
        class _S:
            def invoke(self, prompt):
                return schema(answer="grounded")
        return _S()


def test_chat_complete_returns_text():
    chat = GroqChat(inner=_FakeLC())
    assert chat.complete("hi") == "hello world"


def test_chat_structured_returns_model():
    chat = GroqChat(inner=_FakeLC())
    out = chat.structured("extract", _Schema)
    assert isinstance(out, _Schema)
    assert out.answer == "grounded"
