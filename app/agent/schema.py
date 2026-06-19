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
        props = node.get("properties")
        if node.get("type") == "object" and isinstance(props, dict):
            for name in [k for k in props if k in drop]:
                del props[name]
            node["additionalProperties"] = False
            node["required"] = list(props.keys())
            for sub in props.values():  # recurse into property SCHEMAS only
                _strictify(sub, drop)
        node.pop("default", None)
        # `title` is a JSON-Schema annotation (always a string) — strip it. Do NOT
        # touch a property/def literally NAMED "title" (its value is a schema dict);
        # that's why we never pop keys from the `properties` map above.
        if isinstance(node.get("title"), str):
            node.pop("title", None)
        for key, v in node.items():
            if key == "properties":
                continue  # handled above; its keys are field names, not schemas
            _strictify(v, drop)
    elif isinstance(node, list):
        for v in node:
            _strictify(v, drop)
