from __future__ import annotations

import base64

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
        return self._inner.with_structured_output(schema).invoke(prompt)


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
