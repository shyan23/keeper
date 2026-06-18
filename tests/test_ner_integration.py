"""Extraction node uses an injected EntityExtractor (OpenMed NER) when present,
and is a pure LLM-only no-op when it isn't (no regression, graceful by default)."""
from app.agent.state import Deps, ExtractionResult
from app.agent.nodes.ingest import _extract_one


class _FakeChat:
    def complete(self, prompt):
        return ""

    def structured(self, prompt, schema):
        return ExtractionResult(
            patient_name="John",
            diseases=[{"name": "Hypertension", "confidence": 0.8, "source_span": "HTN"}],
        )


class _FakeNER:
    def extract(self, text):
        return [{"type": "medication", "name": "Metformin", "score": 0.9,
                 "source_span": "Metformin"}]


def test_extract_one_merges_ner_entities_when_extractor_injected():
    deps = Deps(chat=_FakeChat(), vision=None, embedder=None,
                session_factory=None, ner=_FakeNER())
    out = _extract_one(deps, "John, HTN, on Metformin")
    assert "Metformin" in [m["name"] for m in out["medications"]]   # from NER
    assert "Hypertension" in [d["name"] for d in out["diseases"]]    # from LLM, kept


def test_extract_one_is_llm_only_without_ner():
    deps = Deps(chat=_FakeChat(), vision=None, embedder=None, session_factory=None)
    out = _extract_one(deps, "anything")
    assert out["medications"] == []


def test_extract_one_survives_ner_failure():
    class _BoomNER:
        def extract(self, text):
            raise RuntimeError("model not loaded")

    deps = Deps(chat=_FakeChat(), vision=None, embedder=None,
                session_factory=None, ner=_BoomNER())
    out = _extract_one(deps, "x")  # NER blows up -> fall back to LLM-only, no crash
    assert "Hypertension" in [d["name"] for d in out["diseases"]]
