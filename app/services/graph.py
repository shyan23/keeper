"""Medical relationship graph — inference over existing DB data.

Pipeline:
  1. load_patient_data()  – fetch all entities/documents for a patient
  2. build_graph()        – nodes + co-occurrence edges
  3. infer_temporal()     – temporal-proximity edges (±14d)
  4. detect_alerts()      – duplicate tests, dangerous trends, out-of-range repeats
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from sqlalchemy.orm import Session

from app.models import (
    Disease, Document, DocumentEntity, MedicalTest, Medication, TestResult,
)
from app.services import browse as bsvc

# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

_SELF_REFERRED_RE = re.compile(
    r"\b(self[\s-]?refer|routine|follow[\s-]?up|screening)\b", re.I
)


def _doc_meta(db: Session, patient_id: int) -> dict[int, dict]:
    """doc_id → {report_date, doc_type, classification, raw_ocr_text}"""
    rows = (
        db.query(Document)
        .filter(Document.patient_id == patient_id)
        .all()
    )
    return {
        d.id: {
            "report_date": d.report_date,
            "doc_type": d.doc_type or "",
            "classification": d.classification or "",
            "ocr": d.raw_ocr_text or "",
        }
        for d in rows
    }


def _entities_by_doc(db: Session, patient_id: int) -> dict[int, list[dict]]:
    """doc_id → [{entity_type, entity_id, name, confidence}]"""
    rows = (
        db.query(DocumentEntity, Document)
        .join(Document, Document.id == DocumentEntity.document_id)
        .filter(Document.patient_id == patient_id)
        .all()
    )
    result: dict[int, list[dict]] = defaultdict(list)
    for ent, doc in rows:
        result[doc.id].append({
            "entity_type": ent.entity_type,
            "entity_id": ent.entity_id,
            "confidence": ent.confidence or 0.7,
        })
    return result


def _name_map(db: Session, patient_id: int) -> dict[tuple[str, int], str]:
    """(entity_type, entity_id) → human name"""
    m: dict[tuple[str, int], str] = {}
    for d in db.query(Disease).all():
        m[("disease", d.id)] = d.name
    for med in db.query(Medication).all():
        m[("medication", med.id)] = med.name
    for tr, test_name in (
        db.query(TestResult, MedicalTest.name)
        .join(MedicalTest, MedicalTest.id == TestResult.medical_test_id)
        .all()
    ):
        m[("test_result", tr.id)] = test_name
    return m


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

_ENTITY_NODE_TYPE = {
    "disease": "disease",
    "medication": "medication",
    "test_result": "test",
}

_EDGE_TYPE_MAP = {
    ("disease", "medication"): "treated_by",
    ("disease", "test_result"): "monitored_by",
    ("medication", "test_result"): "ordered",
}

# confidence for co-occurrence edges (canonical direction only)
_COOCCUR_CONFIDENCE = {
    ("disease", "medication"): 0.85,
    ("disease", "test_result"): 0.80,
    ("medication", "test_result"): 0.75,
}


def build_graph(db: Session, patient_id: int) -> dict:
    doc_meta = _doc_meta(db, patient_id)
    ents_by_doc = _entities_by_doc(db, patient_id)
    names = _name_map(db, patient_id)
    test_series = bsvc.list_test_results(db, patient_id)

    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    seen_edges: set[tuple] = set()

    def add_node(nid: str, ntype: str, label: str, **kwargs):
        if nid not in nodes:
            nodes[nid] = {"id": nid, "type": ntype, "label": label, **kwargs}

    def add_edge(src: str, dst: str, etype: str, confidence: float):
        key = (src, dst, etype)
        if key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({"from": src, "to": dst, "type": etype, "confidence": round(confidence, 2)})

    # Build nodes from entity mentions
    for doc_id, ents in ents_by_doc.items():
        meta = doc_meta.get(doc_id, {})
        rdate = meta.get("report_date")
        date_str = rdate.strftime("%Y-%m-%d") if rdate else None

        for ent in ents:
            etype = ent["entity_type"]
            eid = ent["entity_id"]
            ntype = _ENTITY_NODE_TYPE.get(etype)
            if not ntype:
                continue
            nid = f"{etype}-{eid}"
            label = names.get((etype, eid), f"{etype}#{eid}")
            add_node(nid, ntype, label, date=date_str)

        # Co-occurrence edges within same document
        relevant = [e for e in ents if e["entity_type"] in _ENTITY_NODE_TYPE]
        for i, a in enumerate(relevant):
            for b in relevant[i + 1:]:
                pair = (a["entity_type"], b["entity_type"])
                rev_pair = (b["entity_type"], a["entity_type"])
                if pair in _EDGE_TYPE_MAP:
                    src_ent, dst_ent = a, b
                    canonical_pair = pair
                elif rev_pair in _EDGE_TYPE_MAP:
                    src_ent, dst_ent = b, a
                    canonical_pair = rev_pair
                else:
                    continue
                conf = _COOCCUR_CONFIDENCE[canonical_pair]
                add_edge(
                    f"{src_ent['entity_type']}-{src_ent['entity_id']}",
                    f"{dst_ent['entity_type']}-{dst_ent['entity_id']}",
                    _EDGE_TYPE_MAP[canonical_pair],
                    conf,
                )

    # Annotate test nodes with latest value & status
    _num_re = re.compile(r"-?\d*\.?\d+")
    _ref_re = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")
    by_test_name: dict[str, list[dict]] = defaultdict(list)
    for row in test_series:
        by_test_name[row["test"].strip().lower()].append(row)

    for nid, node in nodes.items():
        if node["type"] != "test":
            continue
        key = node["label"].strip().lower()
        series = by_test_name.get(key, [])
        if not series:
            continue
        latest = series[0]  # already sorted desc by date
        node["value"] = latest.get("value")
        node["unit"] = latest.get("unit")
        ref = latest.get("reference_range") or ""
        m = _ref_re.search(ref)
        status = "normal"
        if m and latest.get("value"):
            vm = _num_re.match(latest["value"])
            if vm:
                v = float(vm.group(0))
                lo, hi = float(m.group(1)), float(m.group(2))
                if v < lo or v > hi:
                    status = "warning"
                    # check if critical (>20% outside range)
                    if lo and (v < lo * 0.8 or v > hi * 1.2):
                        status = "critical"
        node["status"] = status

    # Temporal proximity edges: prescription docs → lab docs within ±14d
    _infer_temporal(doc_meta, ents_by_doc, edges, seen_edges, nodes)

    alerts = detect_alerts(test_series, doc_meta, ents_by_doc, names)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "alerts": alerts,
    }


def _infer_temporal(
    doc_meta: dict,
    ents_by_doc: dict,
    edges: list,
    seen_edges: set,
    nodes: dict,
):
    """Link medication in prescription doc → test in lab doc if dates ≤14 days apart."""
    prescription_docs: list[tuple[date, int]] = []
    lab_docs: list[tuple[date, int]] = []

    for doc_id, meta in doc_meta.items():
        rdate = meta.get("report_date")
        if not rdate:
            continue
        dt = meta.get("doc_type", "").lower()
        cl = meta.get("classification", "").lower()
        if "prescription" in dt or "prescription" in cl:
            prescription_docs.append((rdate, doc_id))
        elif any(k in dt or k in cl for k in ("lab", "test", "blood", "pathology", "report")):
            lab_docs.append((rdate, doc_id))

    for p_date, p_doc in prescription_docs:
        p_ents = ents_by_doc.get(p_doc, [])
        meds = [e for e in p_ents if e["entity_type"] == "medication"]

        for l_date, l_doc in lab_docs:
            delta = abs((l_date - p_date).days)
            if delta > 60:
                continue  # too far apart, skip even as weak
            confidence = 0.85 if delta <= 14 else 0.45

            # check if self-referred (reduces linkage need)
            ocr = doc_meta[l_doc].get("ocr", "")
            if _SELF_REFERRED_RE.search(ocr):
                continue

            l_ents = ents_by_doc.get(l_doc, [])
            tests = [e for e in l_ents if e["entity_type"] == "test_result"]

            for med in meds:
                for test in tests:
                    src = f"medication-{med['entity_id']}"
                    dst = f"test_result-{test['entity_id']}"
                    if src not in nodes or dst not in nodes:
                        continue
                    key = (src, dst, "ordered")
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    edges.append({
                        "from": src, "to": dst,
                        "type": "ordered",
                        "confidence": round(confidence, 2),
                        "temporal": True,
                        "days_apart": delta,
                    })


# ──────────────────────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d*\.?\d+")
_REF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)")


def _num(v: str | None) -> float | None:
    if not v:
        return None
    m = _NUM_RE.match(v.strip())
    return float(m.group(0)) if m else None


def detect_alerts(
    test_series: list[dict],
    doc_meta: dict,
    ents_by_doc: dict,
    names: dict,
) -> list[dict]:
    alerts: list[dict] = []

    # Group by test name
    by_name: dict[str, list[dict]] = defaultdict(list)
    for row in test_series:
        name = (row.get("test") or "").strip().lower()
        if name:
            by_name[name].append(row)

    for name, rows in by_name.items():
        label = rows[0].get("test", name)
        dated = sorted(
            [r for r in rows if r.get("report_date")],
            key=lambda r: r["report_date"]
        )

        # Duplicate test alert: same test within 30 days
        for i in range(1, len(dated)):
            try:
                d1 = date.fromisoformat(dated[i - 1]["report_date"])
                d2 = date.fromisoformat(dated[i]["report_date"])
                diff = abs((d2 - d1).days)
                if diff <= 30:
                    alerts.append({
                        "type": "duplicate_test",
                        "severity": "warning",
                        "message": f"{label} repeated after only {diff} days",
                        "test": label,
                        "dates": [dated[i - 1]["report_date"], dated[i]["report_date"]],
                    })
            except (ValueError, TypeError):
                pass

        # Dangerous trend: 3 consecutive out-of-range values going same direction
        numeric = [
            {"date": r["report_date"], "value": _num(r.get("value")), "ref": r.get("reference_range")}
            for r in dated if _num(r.get("value")) is not None and r.get("report_date")
        ]
        if len(numeric) >= 3:
            ref_m = _REF_RE.search(numeric[-1]["ref"] or "")
            if ref_m:
                lo, hi = float(ref_m.group(1)), float(ref_m.group(2))
                vals = [n["value"] for n in numeric[-4:]]  # last 4 max
                if all(v is not None for v in vals) and len(vals) >= 3:
                    # all out of range and monotonically worsening
                    out_of_range = [v < lo or v > hi for v in vals]
                    if all(out_of_range[-3:]):
                        diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
                        if all(d > 0 for d in diffs[-2:]):
                            alerts.append({
                                "type": "dangerous_trend",
                                "severity": "critical",
                                "message": f"{label} increasing above normal for {len(vals)} consecutive sessions",
                                "test": label,
                            })
                        elif all(d < 0 for d in diffs[-2:]):
                            alerts.append({
                                "type": "dangerous_trend",
                                "severity": "critical",
                                "message": f"{label} declining below normal for {len(vals)} consecutive sessions",
                                "test": label,
                            })

        # Repeated abnormal: last 2 values out of range
        if len(numeric) >= 2:
            ref_m = _REF_RE.search(numeric[-1]["ref"] or "")
            if ref_m:
                lo, hi = float(ref_m.group(1)), float(ref_m.group(2))
                last2 = numeric[-2:]
                if all((n["value"] < lo or n["value"] > hi) for n in last2):
                    alerts.append({
                        "type": "repeated_abnormal",
                        "severity": "warning",
                        "message": f"{label} out of normal range in last 2 results",
                        "test": label,
                    })

    return alerts
