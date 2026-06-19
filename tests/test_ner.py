"""Deterministic merge of OpenMed NER entities into the LLM extraction.

Pure logic, no model: given the LLM's ExtractionResult dict and a list of NER-found
entities, produce a merged dict. The NER model itself is injected/faked elsewhere.
"""
from app.services.ner import merge_ner_entities


def _base():
    return {
        "patient_name": "John",
        "diseases": [{"name": "Hypertension", "confidence": 0.8, "source_span": "HTN"}],
        "symptoms": [],
        "medications": [],
        "tests": [{"name": "Hb", "value": "13", "confidence": 0.9, "source_span": ""}],
    }


def test_adds_new_ner_disease_with_span_and_confidence():
    ner = [{"type": "disease", "name": "Diabetes Mellitus", "score": 0.95,
            "source_span": "Type 2 DM"}]
    out = merge_ner_entities(_base(), ner)
    names = [d["name"] for d in out["diseases"]]
    assert "Diabetes Mellitus" in names
    added = next(d for d in out["diseases"] if d["name"] == "Diabetes Mellitus")
    assert added["confidence"] == 0.95
    assert added["source_span"] == "Type 2 DM"


def test_dedupes_against_llm_entity_case_insensitive():
    ner = [{"type": "disease", "name": "hypertension", "score": 0.99, "source_span": "x"}]
    out = merge_ner_entities(_base(), ner)
    # LLM's Hypertension is kept; the duplicate NER one is not appended
    assert [d["name"] for d in out["diseases"]] == ["Hypertension"]


def test_routes_medication_and_symptom_to_their_buckets():
    ner = [
        {"type": "medication", "name": "Metformin", "score": 0.9, "source_span": "Metformin"},
        {"type": "symptom", "name": "fatigue", "score": 0.7, "source_span": "tired"},
    ]
    out = merge_ner_entities(_base(), ner)
    assert [m["name"] for m in out["medications"]] == ["Metformin"]
    assert [s["name"] for s in out["symptoms"]] == ["fatigue"]


def test_ignores_unknown_entity_types():
    # NER has no structure for lab values; test_result-like spans must NOT leak in
    ner = [{"type": "test", "name": "RBC", "score": 0.9, "source_span": "RBC 4.5"}]
    out = merge_ner_entities(_base(), ner)
    assert all(d["name"] != "RBC" for d in out["diseases"])
    assert "RBC" not in [t["name"] for t in out["tests"]]  # tests untouched by NER


def test_dedupes_within_ner_list():
    ner = [
        {"type": "disease", "name": "Anemia", "score": 0.9, "source_span": "anemia"},
        {"type": "disease", "name": "anemia", "score": 0.5, "source_span": "anaemia"},
    ]
    out = merge_ner_entities(_base(), ner)
    assert [d["name"] for d in out["diseases"]].count("Anemia") == 1
    assert len([d for d in out["diseases"] if d["name"].lower() == "anemia"]) == 1


def test_does_not_mutate_input():
    base = _base()
    merge_ner_entities(base, [{"type": "disease", "name": "Asthma", "score": 0.8,
                               "source_span": "asthma"}])
    assert [d["name"] for d in base["diseases"]] == ["Hypertension"]
