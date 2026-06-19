from app.services.segment import split_reports, ReportSpec, _ReportSplit


class _FakeChat:
    """Returns a fixed structured split, ignoring the prompt."""
    def __init__(self, split): self._split = split
    def structured(self, prompt, schema): return self._split


def test_split_reports_carries_category():
    split = _ReportSplit(reports=[
        ReportSpec(title="CBC", doc_type="lab report", category="Hematology", pages=[0]),
        ReportSpec(title="Chest X-Ray", doc_type="imaging", category="X-Ray", pages=[1]),
    ])
    out = split_reports(_FakeChat(split), ["cbc page", "xray page"])
    cats = [s["category"] for s in out]
    assert cats == ["Hematology", "X-Ray"]


def test_regex_fallback_category_none():
    # single page -> no LLM call, category must be present and None
    out = split_reports(_FakeChat(_ReportSplit(reports=[])), ["only one page"])
    assert out[0]["category"] is None
