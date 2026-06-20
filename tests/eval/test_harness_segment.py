"""The harness must mirror prod: split a bundle into reports, extract each, union.

A multi-report PDF that holds two dated reports should yield BOTH reports' tests
and the first patient/date — not whatever a single cramped structured() call kept.
No DB, no network: a fake chat returns canned splits/extractions by schema.
"""
from app.eval import harness


class _FakeChat:
    """structured() answers by target schema: _ReportSplit -> two reports (one per
    page); ExtractionResult -> that page's parsed fields."""

    def structured(self, prompt: str, schema, config=None):
        if schema.__name__ == "_ReportSplit":
            return schema(reports=[
                {"title": "CBC", "doc_type": "lab report", "pages": [0],
                 "date": "2023-05-01"},
                {"title": "Lipid", "doc_type": "lab report", "pages": [1],
                 "date": "2023-06-01"},
            ])
        # ExtractionResult — key off page content.
        if "Haemoglobin" in prompt:
            return schema(patient_name="John Akram", patient_age=45,
                          doc_date="2023-05-01",
                          tests=[{"name": "Haemoglobin", "value": "13.5"}])
        return schema(patient_name="John Akram", doc_date="2023-06-01",
                      tests=[{"name": "Cholesterol", "value": "180"}])

    def complete(self, prompt: str, config=None) -> str:  # unused
        return ""


def test_merge_unions_lists_and_takes_first_scalar():
    parts = [
        {"tests": [{"name": "A"}], "patient_name": "John", "doc_date": "d1"},
        {"tests": [{"name": "B"}], "patient_name": "John", "doc_date": "d2"},
    ]
    m = harness._merge_extractions(parts)
    assert [t["name"] for t in m["tests"]] == ["A", "B"]
    assert m["patient_name"] == "John"
    assert m["doc_date"] == "d1"  # first non-empty wins


def test_segment_and_extract_returns_both_reports_tests():
    case = {"pages": ["...Haemoglobin 13.5...", "...Cholesterol 180..."]}
    pred = harness.segment_and_extract(case, _FakeChat())
    names = {t["name"] for t in pred["tests"]}
    assert names == {"Haemoglobin", "Cholesterol"}
    assert pred["patient_name"] == "John Akram"


def test_single_text_case_makes_no_split_call():
    # One page -> split_reports short-circuits (no _ReportSplit call); still extracts.
    case = {"text": "...Haemoglobin 13.5..."}
    pred = harness.segment_and_extract(case, _FakeChat())
    assert [t["name"] for t in pred["tests"]] == ["Haemoglobin"]
