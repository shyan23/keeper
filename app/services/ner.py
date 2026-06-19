"""Hybrid entity extraction: merge OpenMed biomedical NER findings into the LLM
extraction. The LLM gives structure (scalars, test values/units); a dedicated NER
model gives high-precision disease/symptom/medication spans. This merges the two,
keeping the LLM entity on any name clash and adding new NER ones with their span.

`merge_ner_entities` is pure and deterministic — the NER model is injected elsewhere
(see EntityExtractor / OpenMedNER). NER returns no structure for lab values, so test
results are LLM-only and untouched here.
"""
from __future__ import annotations

from typing import Any

# NER entity type -> ExtractionResult bucket. Types outside this map are ignored
# (e.g. a NER "test" span has no value/unit, so it must not pollute tests[]).
_BUCKETS = {"disease": "diseases", "symptom": "symptoms", "medication": "medications"}


def _norm(name: Any) -> str:
    return str(name or "").strip().lower()


def merge_ner_entities(extracted: dict, ner_entities: list[dict]) -> dict:
    """Return a new dict: `extracted` with NER entities unioned into the disease /
    symptom / medication buckets, deduped by normalized name (LLM entity wins)."""
    out = dict(extracted)
    seen: dict[str, set[str]] = {}
    for bucket in _BUCKETS.values():
        items = list(out.get(bucket) or [])
        out[bucket] = items
        seen[bucket] = {_norm(i.get("name")) for i in items}

    for ent in ner_entities:
        bucket = _BUCKETS.get(ent.get("type"))
        if bucket is None:
            continue
        key = _norm(ent.get("name"))
        if not key or key in seen[bucket]:
            continue
        seen[bucket].add(key)
        out[bucket].append({
            "name": ent.get("name"),
            "confidence": ent.get("score", 0.5),
            "source_span": ent.get("source_span", ""),
        })
    return out
