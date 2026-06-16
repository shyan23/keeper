from __future__ import annotations

import base64
import json

from pydantic import BaseModel

from app.config import get_settings


class GroqChat:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            inner = ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0)
        self._inner = inner

    def complete(self, prompt: str) -> str:
        return self._inner.invoke(prompt).content

    def structured(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        # json_object response_format + schema-in-prompt, parsed manually. Unlike
        # tool/function binding on Groq llama (which truncated the tests array to
        # 1 of 25), raw JSON mode returns every item — and in ~3-4s.
        instructed = (
            f"{prompt}\n\nReturn ONLY a JSON object — no prose, no code fences — "
            f"with EVERY item populated, matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        raw = self._inner.bind(
            response_format={"type": "json_object"}
        ).invoke(instructed).content
        return schema.model_validate_json(raw)


class GroqVision:
    def __init__(self, inner=None):
        if inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            inner = ChatGroq(model=s.groq_vision_model, api_key=s.groq_api_key, temperature=0)
        self._inner = inner

    def ocr_image(self, data: bytes, mime: str) -> str:
        b64 = base64.b64encode(data).decode()
        msg = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe ALL text in this medical document verbatim. Output only the text."},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }]
        return self._inner.invoke(msg).content
