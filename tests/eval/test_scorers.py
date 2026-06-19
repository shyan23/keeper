"""Unit tests for the deterministic scorers — the harness's only judge.

These run with no DB, no model, no network: pure functions in, numbers out. They are
the guarantee that the eval scoring itself is correct and reproducible.
"""
from app.eval import scorers as sc


# ---- normalization ----

def test_strip_honorific_and_case():
    assert sc.norm_name("MRS. Nafisa Kabir") == "nafisa kabir"
    assert sc.norm_name("Dr.  John   Akram") == "john akram"
    assert sc.norm_name("Nafisa Kabir") == sc.norm_name("Mrs. Nafisa Kabir")


def test_norm_date_formats_converge():
    assert sc.norm_date("2024-03-12") == "2024-03-12"
    assert sc.norm_date("12/03/2024") == "2024-03-12"   # dd/mm/yyyy
    assert sc.norm_date("12-3-2024") == "2024-03-12"


def test_values_match_numeric_tolerant():
    assert sc.values_match("13.5", "13.5")
    assert sc.values_match("13.50 g/dL", "13.5")        # unit ignored, numeric eq
    assert not sc.values_match("13.5", "14.0")


def test_values_match_text_fallback():
    assert sc.values_match("No active disease", "no active disease")
    assert sc.values_match("clear", "Lung fields clear")  # substring either way


# ---- extraction scoring ----

def test_score_scalars_counts_only_specified_fields():
    pred = {"patient_name": "Mr. John Akram", "patient_age": 45, "doc_type": "CBC"}
    gold = {"patient_name": "John Akram", "patient_age": 45}
    out = sc.score_scalars(pred, gold)
    assert out == {"correct": 2, "total": 2, "misses": []}


def test_score_scalars_flags_mismatch():
    pred = {"patient_name": "Jane Doe", "patient_age": 45}
    gold = {"patient_name": "John Akram", "patient_age": 45}
    out = sc.score_scalars(pred, gold)
    assert out["correct"] == 1 and out["misses"] == ["patient_name"]


def test_score_tests_name_and_value_recall():
    pred = [{"name": "Haemoglobin", "value": "13.5"},
            {"name": "RBC Count", "value": "9.9"}]   # wrong value
    gold = [{"name": "Haemoglobin", "value": "13.5"},
            {"name": "RBC Count", "value": "4.8"},
            {"name": "WBC Count", "value": "7200"}]   # missing entirely
    out = sc.score_tests(pred, gold)
    assert out["expected"] == 3
    assert out["name_matched"] == 2     # Hb + RBC found by name
    assert out["value_matched"] == 1    # only Hb value correct


def test_score_entities_recall():
    pred = [{"name": "Hypertension"}]
    gold = [{"name": "Hypertension"}, {"name": "Type 2 Diabetes Mellitus"}]
    out = sc.score_entities(pred, gold)
    assert out["matched"] == 1 and out["expected"] == 2
    assert out["missed"] == ["type 2 diabetes mellitus"]


def test_score_extraction_shape():
    pred = {"patient_name": "John Akram", "tests": [{"name": "Hb", "value": "13"}],
            "diseases": [], "symptoms": [], "medications": []}
    gold = {"patient_name": "John Akram", "tests": [{"name": "Hb", "value": "13"}]}
    out = sc.score_extraction(pred, gold)
    assert out["scalars"]["correct"] == 1
    assert out["tests"]["value_matched"] == 1
    assert set(out["entities"]) == {"diseases", "symptoms", "medications"}


# ---- retrieval scoring ----

def test_recall_at_k_by_original_name():
    hits = [{"original_name": "lipid_jan.pdf"}, {"original_name": "cbc_jun.pdf"}]
    assert sc.recall_at_k(hits, "cbc_jun.pdf")
    assert not sc.recall_at_k(hits, "vitd_feb.pdf")


def test_recall_at_k_by_doc_type():
    hits = [{"doc_type": "CBC", "original_name": None}]
    assert sc.recall_at_k(hits, "cbc")


def test_answer_contains():
    assert sc.answer_contains("Her hemoglobin is 12.1 g/dL.", "12.1")
    assert not sc.answer_contains("No data available.", "12.1")
