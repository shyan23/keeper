"""Tests for app/services/graph.py.

Unit tests for detect_alerts() use plain Python dicts (no DB needed).
Integration test for _infer_temporal via build_graph() uses the test DB.
"""
import datetime as dt

from app.services.graph import detect_alerts, build_graph
from app.models import (
    Disease, Document, DocumentEntity, MedicalTest,
    Medication, Patient, TestResult,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers for unit tests (no DB)
# ──────────────────────────────────────────────────────────────────────

def _make_series(entries):
    """Build a test_series list from (date_str, value, ref) tuples."""
    return [
        {
            "test": "HbA1c",
            "value": value,
            "unit": "%",
            "reference_range": ref,
            "report_date": date_str,
        }
        for date_str, value, ref in entries
    ]


# ──────────────────────────────────────────────────────────────────────
# Unit tests for detect_alerts()
# ──────────────────────────────────────────────────────────────────────

def test_detect_alerts_duplicate_test_within_30_days():
    """duplicate_test alert fires when same test repeated within 30 days."""
    series = _make_series([
        ("2024-01-01", "7.2", "4.0-6.5"),
        ("2024-01-20", "7.5", "4.0-6.5"),  # 19 days later — should trigger
    ])
    alerts = detect_alerts(series, {}, {}, {})
    dup = [a for a in alerts if a["type"] == "duplicate_test"]
    assert len(dup) >= 1
    assert dup[0]["severity"] == "warning"
    assert dup[0]["test"] == "HbA1c"
    assert dup[0]["dates"] == ["2024-01-01", "2024-01-20"]


def test_detect_alerts_no_duplicate_beyond_30_days():
    """No duplicate_test alert when tests are more than 30 days apart."""
    series = _make_series([
        ("2024-01-01", "7.2", "4.0-6.5"),
        ("2024-02-15", "7.5", "4.0-6.5"),  # 45 days later — should NOT trigger
    ])
    alerts = detect_alerts(series, {}, {}, {})
    dup = [a for a in alerts if a["type"] == "duplicate_test"]
    assert len(dup) == 0


def test_detect_alerts_dangerous_trend_increasing():
    """dangerous_trend fires for 3 consecutive out-of-range values moving upward."""
    series = _make_series([
        ("2024-01-01", "7.0", "4.0-6.5"),
        ("2024-02-01", "7.5", "4.0-6.5"),
        ("2024-03-01", "8.0", "4.0-6.5"),
    ])
    alerts = detect_alerts(series, {}, {}, {})
    trend = [a for a in alerts if a["type"] == "dangerous_trend"]
    assert len(trend) >= 1
    assert trend[0]["severity"] == "critical"
    assert trend[0]["test"] == "HbA1c"


def test_detect_alerts_dangerous_trend_decreasing():
    """dangerous_trend fires for 3 consecutive out-of-range values moving downward."""
    series = _make_series([
        ("2024-01-01", "2.0", "4.0-6.5"),
        ("2024-02-01", "1.5", "4.0-6.5"),
        ("2024-03-01", "1.0", "4.0-6.5"),
    ])
    alerts = detect_alerts(series, {}, {}, {})
    trend = [a for a in alerts if a["type"] == "dangerous_trend"]
    assert len(trend) >= 1
    assert trend[0]["severity"] == "critical"


def test_detect_alerts_dangerous_trend_not_monotone():
    """No dangerous_trend when values are out-of-range but not monotonically moving."""
    series = _make_series([
        ("2024-01-01", "7.5", "4.0-6.5"),
        ("2024-02-01", "8.0", "4.0-6.5"),
        ("2024-03-01", "7.2", "4.0-6.5"),  # went down — breaks the trend
    ])
    alerts = detect_alerts(series, {}, {}, {})
    trend = [a for a in alerts if a["type"] == "dangerous_trend"]
    assert len(trend) == 0


def test_detect_alerts_repeated_abnormal_last_2():
    """repeated_abnormal fires when last 2 results are out of reference range."""
    series = _make_series([
        ("2024-01-01", "5.0", "4.0-6.5"),  # in range
        ("2024-03-01", "7.5", "4.0-6.5"),  # out of range
        ("2024-06-01", "8.0", "4.0-6.5"),  # out of range — 2 consecutive
    ])
    alerts = detect_alerts(series, {}, {}, {})
    abnormal = [a for a in alerts if a["type"] == "repeated_abnormal"]
    assert len(abnormal) >= 1
    assert abnormal[0]["severity"] == "warning"
    assert abnormal[0]["test"] == "HbA1c"


def test_detect_alerts_repeated_abnormal_not_fired_when_last_in_range():
    """repeated_abnormal does NOT fire when the most recent value is in range."""
    series = _make_series([
        ("2024-01-01", "7.5", "4.0-6.5"),  # out of range
        ("2024-06-01", "5.0", "4.0-6.5"),  # in range — clears it
    ])
    alerts = detect_alerts(series, {}, {}, {})
    abnormal = [a for a in alerts if a["type"] == "repeated_abnormal"]
    assert len(abnormal) == 0


# ──────────────────────────────────────────────────────────────────────
# Integration test: _infer_temporal via build_graph()
# ──────────────────────────────────────────────────────────────────────

def _seed_temporal(db):
    """
    Seed: prescription doc (Jan 1) with Metformin + lab doc (Jan 8) with HbA1c.
    Gap = 7 days → should produce a temporal 'ordered' edge with confidence 0.85.
    Returns (patient_id, med_entity_id, test_result_entity_id).
    """
    patient = Patient(name="Graph Temporal Test")
    db.add(patient)
    db.flush()

    # Prescription document
    presc_doc = Document(
        patient_id=patient.id,
        doc_type="prescription",
        classification="prescription",
        report_date=dt.date(2024, 1, 1),
        raw_ocr_text="Prescribed Metformin 500mg",
    )
    db.add(presc_doc)
    db.flush()

    # Medication entity
    med = Medication(name="Metformin")
    db.add(med)
    db.flush()

    med_link = DocumentEntity(
        document_id=presc_doc.id,
        entity_type="medication",
        entity_id=med.id,
        confidence=0.90,
    )
    db.add(med_link)
    db.flush()

    # Lab document (7 days later)
    lab_doc = Document(
        patient_id=patient.id,
        doc_type="lab report",
        classification="lab",
        report_date=dt.date(2024, 1, 8),
        raw_ocr_text="HbA1c result: 7.2%",
    )
    db.add(lab_doc)
    db.flush()

    # MedicalTest + TestResult
    mt = MedicalTest(name="HbA1c")
    db.add(mt)
    db.flush()

    tr = TestResult(
        medical_test_id=mt.id,
        value="7.2",
        unit="%",
        reference_range="4.0-6.5",
    )
    db.add(tr)
    db.flush()

    tr_link = DocumentEntity(
        document_id=lab_doc.id,
        entity_type="test_result",
        entity_id=tr.id,
        confidence=0.90,
    )
    db.add(tr_link)
    db.commit()

    return patient.id, med.id, tr.id


def test_build_graph_temporal_edge(db):
    """
    build_graph() produces a temporal 'ordered' edge (confidence 0.85, days_apart=7)
    from a prescription doc (Metformin) to a lab doc (HbA1c) 7 days apart.
    """
    patient_id, med_id, tr_id = _seed_temporal(db)

    result = build_graph(db, patient_id)

    nodes_by_id = {n["id"]: n for n in result["nodes"]}
    med_nid = f"medication-{med_id}"
    tr_nid = f"test_result-{tr_id}"

    assert med_nid in nodes_by_id, "Medication node missing"
    assert tr_nid in nodes_by_id, "TestResult node missing"

    temporal_edges = [
        e for e in result["edges"]
        if e.get("temporal") is True
        and e["from"] == med_nid
        and e["to"] == tr_nid
        and e["type"] == "temporally_ordered"
    ]
    assert len(temporal_edges) == 1, (
        f"Expected 1 temporal 'temporally_ordered' edge, got {len(temporal_edges)}. "
        f"All edges: {result['edges']}"
    )
    edge = temporal_edges[0]
    assert edge["confidence"] == 0.85
    assert edge["days_apart"] == 7


def test_build_graph_temporal_edge_skipped_beyond_60_days(db):
    """No temporal edge when prescription and lab doc are more than 60 days apart."""
    patient = Patient(name="Graph Temporal Far")
    db.add(patient)
    db.flush()

    presc_doc = Document(
        patient_id=patient.id,
        doc_type="prescription",
        report_date=dt.date(2024, 1, 1),
    )
    db.add(presc_doc)
    db.flush()

    med = Medication(name="Aspirin")
    db.add(med)
    db.flush()
    db.add(DocumentEntity(
        document_id=presc_doc.id, entity_type="medication",
        entity_id=med.id, confidence=0.9,
    ))
    db.flush()

    lab_doc = Document(
        patient_id=patient.id,
        doc_type="lab report",
        report_date=dt.date(2024, 3, 15),  # 74 days later
    )
    db.add(lab_doc)
    db.flush()

    mt = MedicalTest(name="CBC")
    db.add(mt)
    db.flush()

    tr = TestResult(medical_test_id=mt.id, value="14.0", unit="g/dL", reference_range="12-16")
    db.add(tr)
    db.flush()
    db.add(DocumentEntity(
        document_id=lab_doc.id, entity_type="test_result",
        entity_id=tr.id, confidence=0.9,
    ))
    db.commit()

    result = build_graph(db, patient.id)
    temporal_edges = [e for e in result["edges"] if e.get("temporal") is True]
    assert len(temporal_edges) == 0


def test_build_graph_temporal_edge_skipped_self_referred(db):
    """No temporal edge when lab doc OCR contains a self-referral marker."""
    patient = Patient(name="Graph Temporal Self")
    db.add(patient)
    db.flush()

    presc_doc = Document(
        patient_id=patient.id,
        doc_type="prescription",
        report_date=dt.date(2024, 1, 1),
    )
    db.add(presc_doc)
    db.flush()

    med = Medication(name="Lisinopril")
    db.add(med)
    db.flush()
    db.add(DocumentEntity(
        document_id=presc_doc.id, entity_type="medication",
        entity_id=med.id, confidence=0.9,
    ))
    db.flush()

    lab_doc = Document(
        patient_id=patient.id,
        doc_type="lab report",
        report_date=dt.date(2024, 1, 7),
        raw_ocr_text="This is a routine screening test.",  # matches skip regex
    )
    db.add(lab_doc)
    db.flush()

    mt = MedicalTest(name="Lipid Panel")
    db.add(mt)
    db.flush()

    tr = TestResult(medical_test_id=mt.id, value="200", unit="mg/dL", reference_range="0-200")
    db.add(tr)
    db.flush()
    db.add(DocumentEntity(
        document_id=lab_doc.id, entity_type="test_result",
        entity_id=tr.id, confidence=0.9,
    ))
    db.commit()

    result = build_graph(db, patient.id)
    temporal_edges = [e for e in result["edges"] if e.get("temporal") is True]
    assert len(temporal_edges) == 0
