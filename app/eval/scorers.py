"""Deterministic scorers — the only "judge" in the harness.

Pure functions, no I/O, no LLM. Given a model's prediction and the golden answer,
each returns a number or bool by exact / numeric / set-overlap comparison. Because
they're pure, they're unit-tested directly (tests/eval/test_scorers.py) and a whole
eval run is reproducible with no API calls.
"""
from __future__ import annotations

import re
from typing import Any

_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "md", "prof", "mr.", "mrs.",
               "ms.", "dr.", "prof.", "sri", "smt"}


def normalize(s: Any) -> str:
    """Lowercase, strip surrounding punctuation, collapse internal whitespace."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .,:;-_/")


def strip_honorific(name: Any) -> str:
    toks = normalize(name).split()
    while toks and toks[0].strip(".") in {h.strip(".") for h in _HONORIFICS}:
        toks = toks[1:]
    return " ".join(toks)


def norm_name(s: Any) -> str:
    return strip_honorific(s)


def norm_date(s: Any) -> str:
    """Best-effort normalize to ISO YYYY-MM-DD; fall back to normalized string."""
    raw = normalize(s)
    if not raw:
        return ""
    m = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        y, mo, d = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    # dd/mm/yyyy or dd-mm-yyyy
    m = re.match(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})", raw)
    if m:
        d, mo, y = m.groups()
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d}"
    return raw


def _as_float(s: Any) -> float | None:
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(s))
    return float(m.group()) if m else None


def values_match(pred: Any, gold: Any, *, rel_tol: float = 1e-3) -> bool:
    """Numeric-aware value equality. If both look numeric, compare with tolerance;
    otherwise fall back to normalized-substring match (handles units/wording)."""
    pg, gg = _as_float(pred), _as_float(gold)
    if pg is not None and gg is not None:
        denom = max(abs(gg), 1e-9)
        return abs(pg - gg) / denom <= rel_tol
    p, g = normalize(pred), normalize(gold)
    if not g:
        return not p
    return g in p or p in g


# ---- extraction ----

_SCALARS = ("patient_name", "patient_age", "patient_gender", "doc_type",
            "doc_date", "doctor")


def _scalar_match(field: str, pred: Any, gold: Any) -> bool:
    if field == "patient_name" or field == "doctor":
        return norm_name(pred) == norm_name(gold)
    if field == "doc_date":
        return norm_date(pred) == norm_date(gold)
    if field == "patient_age":
        return _as_float(pred) == _as_float(gold)
    return normalize(pred) == normalize(gold)


def score_scalars(pred: dict, gold: dict) -> dict:
    """Per-scalar-field accuracy over only the fields the golden case specifies."""
    fields = [f for f in _SCALARS if f in gold and gold[f] is not None]
    hits = [f for f in fields if _scalar_match(f, pred.get(f), gold[f])]
    misses = [f for f in fields if f not in hits]
    return {"correct": len(hits), "total": len(fields), "misses": misses}


def _names(items: list[dict] | None) -> set[str]:
    return {norm_name(i.get("name")) for i in (items or []) if i.get("name")}


def score_entities(pred_items: list[dict] | None, gold_items: list[dict] | None) -> dict:
    """Recall of expected entity names (diseases/symptoms/medications)."""
    gold = _names(gold_items)
    if not gold:
        return {"matched": 0, "expected": 0, "missed": []}
    pred = _names(pred_items)
    matched = gold & pred
    return {"matched": len(matched), "expected": len(gold),
            "missed": sorted(gold - pred)}


def score_tests(pred_tests: list[dict] | None, gold_tests: list[dict] | None) -> dict:
    """For each expected test: name found? value correct? Reports both recalls."""
    gold = gold_tests or []
    if not gold:
        return {"name_matched": 0, "value_matched": 0, "expected": 0, "missed": []}
    pred = pred_tests or []
    by_name: dict[str, dict] = {norm_name(t.get("name")): t for t in pred}
    name_hits = value_hits = 0
    missed: list[str] = []
    for g in gold:
        gname = norm_name(g.get("name"))
        p = by_name.get(gname)
        if p is None:
            missed.append(g.get("name", gname))
            continue
        name_hits += 1
        if "value" not in g or g.get("value") is None or values_match(p.get("value"), g["value"]):
            value_hits += 1
        else:
            missed.append(f"{g.get('name')} (value)")
    return {"name_matched": name_hits, "value_matched": value_hits,
            "expected": len(gold), "missed": missed}


def score_extraction(pred: dict, gold: dict) -> dict:
    """Full per-case extraction scorecard. `pred` = ExtractionResult.model_dump()."""
    scal = score_scalars(pred, gold)
    tests = score_tests(pred.get("tests"), gold.get("tests"))
    ent = {k: score_entities(pred.get(k), gold.get(k))
           for k in ("diseases", "symptoms", "medications")}
    return {"scalars": scal, "tests": tests, "entities": ent}


# ---- retrieval ----

def recall_at_k(retrieved: list[dict], expected_doc: str) -> bool:
    """Did a chunk from the expected document appear in the top-k hits?
    Matches on original_name first, then doc_type (whichever the golden case gives)."""
    want = normalize(expected_doc)
    for h in retrieved:
        if normalize(h.get("original_name")) == want or normalize(h.get("doc_type")) == want:
            return True
    return False


def answer_contains(answer: str, needle: str) -> bool:
    return normalize(needle) in normalize(answer)
