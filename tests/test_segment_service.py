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


def test_orphan_page_recovered_as_prescription():
    # LLM names a report for page 0 only; page 1 (a prescription) is ignored. It
    # must still surface as its own report, typed as a prescription, not dropped.
    split = _ReportSplit(reports=[
        ReportSpec(title="CBC", doc_type="lab report", pages=[0]),
    ])
    out = split_reports(_FakeChat(split),
                        ["cbc results", "Dr. Alam MBBS\nTab. Yamadin 20\nCap. Acteria"])
    assert sorted(i for s in out for i in s["pages"]) == [0, 1]   # nothing dropped
    rx = next(s for s in out if s["pages"] == [1])
    assert rx["doc_type"] == "prescription"


def test_blank_orphan_page_not_recovered():
    # An empty page the LLM skipped should stay skipped (no junk report).
    split = _ReportSplit(reports=[ReportSpec(title="CBC", pages=[0])])
    out = split_reports(_FakeChat(split), ["cbc results", "   "])
    assert [s["pages"] for s in out] == [[0]]
