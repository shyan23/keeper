"""Trend series for numeric test results — the single data source of truth for
the Trends tab (Chart.js now) and the future PDF report (matplotlib). Pure
functions over a DB session; no HTTP, no rendering.

A "metric" is a test name normalized to lower/trimmed. Only metrics with >=2
numeric, dated data points are exposed (a single point is not a trend).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from sqlalchemy.orm import Session

from app.services import browse as bsvc

# leading signed float, e.g. "11.2", "-0.5", ".01"
_NUM_RE = re.compile(r"-?\d*\.?\d+")
# "low-high" interval, e.g. "12-16", "12.0 – 16.0"
_REF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")


def _num(value: str | None) -> float | None:
    if not value:
        return None
    m = _NUM_RE.match(value.strip())
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _ref(reference: str | None) -> tuple[float | None, float | None]:
    if not reference:
        return None, None
    m = _REF_RE.search(reference)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _key(name: str) -> str:
    return (name or "").strip().lower()


def list_metrics(db: Session, patient_id: int) -> list[dict]:
    """[{key, label, unit, n}] for tests with >=2 numeric, dated points."""
    rows = bsvc.list_test_results(db, patient_id=patient_id)
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if _num(r.get("value")) is None or not r.get("report_date"):
            continue
        by_key[_key(r.get("test") or "")].append(r)
    metrics = []
    for key, items in by_key.items():
        if not key or len(items) < 2:
            continue
        label = Counter(r["test"] for r in items if r.get("test")).most_common(1)[0][0]
        unit = next((r["unit"] for r in items if r.get("unit")), "")
        metrics.append({"key": key, "label": label, "unit": unit, "n": len(items)})
    metrics.sort(key=lambda m: m["label"].lower())
    return metrics


def metric_series(db: Session, patient_id: int, key: str) -> dict:
    """{key,label,unit,ref_low,ref_high,points:[{date,value,in_range}]} for one
    metric. Points: numeric + dated only, sorted ascending by date."""
    rows = [r for r in bsvc.list_test_results(db, patient_id=patient_id)
            if _key(r.get("test") or "") == _key(key)]
    label = next((r["test"] for r in rows if r.get("test")), key)
    unit = next((r["unit"] for r in rows if r.get("unit")), "")
    ref_low = ref_high = None
    for r in rows:
        lo, hi = _ref(r.get("reference") or r.get("reference_range"))
        if lo is not None:
            ref_low, ref_high = lo, hi
            break
    points = []
    for r in rows:
        v = _num(r.get("value"))
        if v is None or not r.get("report_date"):
            continue
        in_range = True
        if ref_low is not None and ref_high is not None:
            in_range = ref_low <= v <= ref_high
        points.append({"date": r["report_date"], "value": v, "in_range": in_range})
    points.sort(key=lambda p: p["date"])
    return {"key": _key(key), "label": label, "unit": unit,
            "ref_low": ref_low, "ref_high": ref_high, "points": points}
