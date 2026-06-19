from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ValidationError


def validate_and_retry(invoke_raw: Callable[[str], str],
                       schema: type[BaseModel],
                       attempts: int = 3) -> BaseModel:
    """Parse+validate raw JSON; on ValidationError, re-call invoke_raw with the
    error appended so the model can self-correct. Bounded. Provider-independent.

    invoke_raw(extra_instruction) -> raw JSON string. `extra_instruction` is ""
    on the first call, then the validation error on retries.
    """
    last: Exception | None = None
    extra = ""
    for _ in range(attempts):
        raw = invoke_raw(extra)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last = e
            extra = (f"\n\nYour previous output was INVALID:\n{e}\n"
                     f"Return corrected JSON matching the schema exactly — no prose.")
    raise last  # type: ignore[misc]
