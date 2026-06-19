import pytest
from pydantic import BaseModel
from app.agent.structured import validate_and_retry


class _Schema(BaseModel):
    answer: str


def test_returns_validated_model_first_try():
    out = validate_and_retry(lambda extra: '{"answer": "ok"}', _Schema)
    assert isinstance(out, _Schema) and out.answer == "ok"


def test_retries_then_succeeds_passing_error_back():
    seen = []

    def invoke_raw(extra: str) -> str:
        seen.append(extra)
        return "not json" if len(seen) == 1 else '{"answer": "fixed"}'

    out = validate_and_retry(invoke_raw, _Schema)
    assert out.answer == "fixed"
    assert len(seen) == 2
    assert seen[0] == ""              # first call: no error context
    assert "INVALID" in seen[1]       # retry call: error fed back


def test_raises_after_exhausting_attempts():
    with pytest.raises(Exception):
        validate_and_retry(lambda extra: "still not json", _Schema, attempts=2)
