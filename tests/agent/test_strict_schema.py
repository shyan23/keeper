from app.agent.schema import to_strict_schema
from app.agent.state import ExtractionResult


def test_top_level_object_is_strict():
    s = to_strict_schema(ExtractionResult)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == set(s["properties"].keys())


def test_no_defaults_or_titles_remain():
    s = to_strict_schema(ExtractionResult)

    def walk(node):
        if isinstance(node, dict):
            assert "default" not in node
            assert "title" not in node
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(s)


def test_drop_removes_fields_at_every_level():
    s = to_strict_schema(ExtractionResult, drop={"source_span", "confidence"})
    # root level
    assert "source_span" not in s["properties"]
    assert "confidence" not in s["properties"]
    assert "source_span" not in s["required"]
    # nested defs (ExtractedTest / ExtractedEntity)
    for sub in s["$defs"].values():
        if sub.get("type") == "object":
            assert "source_span" not in sub["properties"]
            assert "confidence" not in sub["properties"]
            assert set(sub["required"]) == set(sub["properties"].keys())


def test_nested_defs_are_strictified():
    # ExtractedTest / ExtractedEntity live under $defs and must also be strict.
    s = to_strict_schema(ExtractionResult)
    assert "$defs" in s
    for sub in s["$defs"].values():
        if sub.get("type") == "object":
            assert sub["additionalProperties"] is False
            assert set(sub["required"]) == set(sub["properties"].keys())
