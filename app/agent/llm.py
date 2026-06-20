from __future__ import annotations

import base64
import json

from pydantic import BaseModel

from app.config import get_settings


class GroqChat:
    # Strict json_schema (constrained decoding) is only available on these Groq
    # models; everything else falls back to best-effort json_object + retry.
    STRICT_MODELS = {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}

    def __init__(self, inner=None, structured_inner=None):
        # Both clients are lazy: complete()-only callers never build the gpt-oss
        # structured client, and vice-versa.
        self._inner = inner
        self._structured_inner = structured_inner

    def _chat_client(self):
        if self._inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            self._inner = ChatGroq(model=s.groq_model, api_key=s.groq_api_key,
                                   temperature=0)
        return self._inner

    def _structured_client(self):
        if self._structured_inner is None:
            from langchain_groq import ChatGroq
            s = get_settings()
            # max_tokens is RESERVED against the free-tier 8000 TPM budget, not
            # just billed on use — 8192 alone exceeds the limit, so every call
            # 413s. 4096 fits (input here is <2k tok) and is ample now that the
            # output-bloating source_span/confidence fields are dropped.
            self._structured_inner = ChatGroq(model=s.groq_structured_model,
                                              api_key=s.groq_api_key, temperature=0,
                                              max_tokens=4096)
        return self._structured_inner

    def complete(self, prompt: str, config: dict | None = None) -> str:
        return self._chat_client().invoke(prompt, config=config).content

    def structured(self, prompt: str, schema: type[BaseModel], config: dict | None = None) -> BaseModel:
        from app.agent.schema import to_strict_schema
        from app.agent.structured import validate_and_retry

        model = get_settings().groq_structured_model
        client = self._structured_client()

        if model in self.STRICT_MODELS:
            # Constrained decoding. Drop defaulted metadata fields: source_span
            # echoes document text per item, and being strict-required it bloats
            # output until long multi-report docs exceed the token budget (-> 400).
            # No-op for schemas without these fields. Pydantic restores defaults.
            rf = {"type": "json_schema",
                  "json_schema": {"name": schema.__name__, "strict": True,
                                  "schema": to_strict_schema(
                                      schema, drop={"source_span"})}}
            raw = client.bind(response_format=rf).invoke(prompt, config=config).content
            return schema.model_validate_json(raw)

        # Best-effort json_object (e.g. llama-3.3) + self-correcting retry.
        base = (f"{prompt}\n\nReturn ONLY a JSON object — no prose, no code fences — "
                f"matching this JSON schema:\n{json.dumps(schema.model_json_schema())}")

        def invoke_raw(extra: str) -> str:
            return client.bind(
                response_format={"type": "json_object"}
            ).invoke(base + extra, config=config).content

        return validate_and_retry(invoke_raw, schema)


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
