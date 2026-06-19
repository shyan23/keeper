from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel


def to_strict_schema(model: type[BaseModel], drop: Iterable[str] = ()) -> dict:
    """Pydantic JSON schema -> OpenAI strict-mode subset (Groq json_schema strict:true).

    Strict mode requires every object to list ALL properties in `required`, set
    `additionalProperties: false`, and omit `default`. Nested models live under
    `$defs`; `$ref` is preserved (strict mode resolves it).

    `drop` removes named properties at every object level. Use it for defaulted
    metadata fields (e.g. source_span, confidence) that bloat output and, being
    strict-required, make long generations exceed the token budget — Pydantic
    restores them from their defaults after parsing."""
    schema = model.model_json_schema()
    _strictify(schema, frozenset(drop))
    return schema


def _strictify(node, drop: frozenset[str]) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            for name in [k for k in node["properties"] if k in drop]:
                del node["properties"][name]
            node["additionalProperties"] = False
            node["required"] = list(node["properties"].keys())
        node.pop("default", None)
        node.pop("title", None)
        for v in node.values():
            _strictify(v, drop)
    elif isinstance(node, list):
        for v in node:
            _strictify(v, drop)
