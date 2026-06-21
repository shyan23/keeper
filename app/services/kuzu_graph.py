"""Kuzu embedded graph DB for per-patient medical relationship graphs.

Write: ingest(patient_id, graph_dict)  — full rebuild from build_graph() output
Read:  query_subgraph(patient_id, entity_name) → {nodes, edges, alerts:[]}

Each patient gets its own directory: data/kuzu_db/{patient_id}/
Rebuild is atomic: rmtree then fresh insert. Non-blocking (called from thread).
"""
from __future__ import annotations

import logging
import shutil
import threading
from pathlib import Path

log = logging.getLogger(__name__)

KUZU_DIR = Path("data/kuzu_db")

_NODE_TABLES: dict[str, str] = {
    "disease": "Disease",
    "medication": "Medication",
    "test": "TestResult",
}
_TABLE_TO_TYPE = {v: k for k, v in _NODE_TABLES.items()}

_EDGE_META: dict[str, tuple[str, str, str]] = {
    "treated_by":        ("TREATED_BY",        "Disease",    "Medication"),
    "monitored_by":      ("MONITORED_BY",       "Disease",    "TestResult"),
    "ordered":           ("ORDERED",            "Medication", "TestResult"),
    "temporally_ordered":("TEMPORALLY_ORDERED", "Medication", "TestResult"),
}

_SCHEMA_STMTS = [
    "CREATE NODE TABLE IF NOT EXISTS Disease(id STRING, label STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS Medication(id STRING, label STRING, PRIMARY KEY(id))",
    "CREATE NODE TABLE IF NOT EXISTS TestResult(id STRING, label STRING, value STRING, unit STRING, status STRING, PRIMARY KEY(id))",
    "CREATE REL TABLE IF NOT EXISTS TREATED_BY(FROM Disease TO Medication, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS MONITORED_BY(FROM Disease TO TestResult, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS ORDERED(FROM Medication TO TestResult, confidence DOUBLE)",
    "CREATE REL TABLE IF NOT EXISTS TEMPORALLY_ORDERED(FROM Medication TO TestResult, confidence DOUBLE, days_apart INT64)",
]

# Per-patient write lock so two simultaneous ingests can't corrupt the same dir.
_write_locks: dict[int, threading.Lock] = {}
_lock_map_lock = threading.Lock()


def _patient_lock(patient_id: int) -> threading.Lock:
    with _lock_map_lock:
        if patient_id not in _write_locks:
            _write_locks[patient_id] = threading.Lock()
        return _write_locks[patient_id]


def _open_conn(patient_id: int):
    import kuzu
    KUZU_DIR.mkdir(parents=True, exist_ok=True)  # ensure parent dir exists; Kuzu creates the DB subdir
    path = KUZU_DIR / str(patient_id)
    db = kuzu.Database(str(path))
    conn = kuzu.Connection(db)
    for stmt in _SCHEMA_STMTS:
        try:
            conn.execute(stmt)
        except Exception:
            pass  # table already exists
    return conn


def ingest(patient_id: int, graph: dict) -> None:
    """Rebuild Kuzu graph for patient from build_graph() output. Thread-safe."""
    lock = _patient_lock(patient_id)
    if not lock.acquire(blocking=False):
        log.debug("kuzu ingest skipped (another ingest in progress for patient %s)", patient_id)
        return
    try:
        path = KUZU_DIR / str(patient_id)
        if path.exists():
            shutil.rmtree(path)
        conn = _open_conn(patient_id)
        _insert_nodes(conn, graph.get("nodes", []))
        _insert_edges(conn, graph.get("edges", []))
        log.info("kuzu: patient %s — %d nodes, %d edges",
                 patient_id, len(graph.get("nodes", [])), len(graph.get("edges", [])))
    except Exception as e:
        log.warning("kuzu ingest failed (patient %s): %s", patient_id, e)
    finally:
        lock.release()


def _insert_nodes(conn, nodes: list[dict]) -> None:
    for node in nodes:
        table = _NODE_TABLES.get(node["type"])
        if not table:
            continue
        nid, label = node["id"], node.get("label", "")
        try:
            if table == "TestResult":
                conn.execute(
                    "CREATE (:TestResult {id: $id, label: $label, value: $value, unit: $unit, status: $status})",
                    {"id": nid, "label": label,
                     "value": str(node.get("value") or ""),
                     "unit": str(node.get("unit") or ""),
                     "status": node.get("status") or "normal"},
                )
            else:
                conn.execute(
                    f"CREATE (:{table} {{id: $id, label: $label}})",
                    {"id": nid, "label": label},
                )
        except Exception as e:
            log.debug("kuzu node skip (%s): %s", nid, e)


def _insert_edges(conn, edges: list[dict]) -> None:
    for edge in edges:
        meta = _EDGE_META.get(edge["type"])
        if not meta:
            continue
        rel, src_table, dst_table = meta
        conf = float(edge.get("confidence", 0.5))
        try:
            if edge["type"] == "temporally_ordered":
                conn.execute(
                    f"MATCH (a:{src_table} {{id: $src}}), (b:{dst_table} {{id: $dst}}) "
                    f"CREATE (a)-[:{rel} {{confidence: $conf, days_apart: $days}}]->(b)",
                    {"src": edge["from"], "dst": edge["to"],
                     "conf": conf, "days": int(edge.get("days_apart", 0))},
                )
            else:
                conn.execute(
                    f"MATCH (a:{src_table} {{id: $src}}), (b:{dst_table} {{id: $dst}}) "
                    f"CREATE (a)-[:{rel} {{confidence: $conf}}]->(b)",
                    {"src": edge["from"], "dst": edge["to"], "conf": conf},
                )
        except Exception as e:
            log.debug("kuzu edge skip (%s→%s): %s", edge.get("from"), edge.get("to"), e)


def query_subgraph(patient_id: int, entity_name: str) -> dict:
    """1-hop subgraph around entity_name. Returns {nodes, edges, alerts:[]}."""
    try:
        conn = _open_conn(patient_id)
    except Exception as e:
        log.warning("kuzu open failed (patient %s): %s", patient_id, e)
        return {"nodes": [], "edges": [], "alerts": []}

    name_lower = entity_name.lower().strip()
    nodes: dict[str, dict] = {}

    # Find seed nodes matching entity_name
    for node_type, table in _NODE_TABLES.items():
        try:
            res = conn.execute(
                f"MATCH (n:{table}) WHERE lower(n.label) CONTAINS $name RETURN n.id, n.label",
                {"name": name_lower},
            )
            while res.has_next():
                nid, label = res.get_next()
                nodes[nid] = {"id": nid, "type": node_type, "label": label, "date": None, "status": "normal"}
        except Exception as e:
            log.debug("kuzu seed query (%s): %s", table, e)

    if not nodes:
        return {"nodes": [], "edges": [], "alerts": []}

    edges: list[dict] = []
    seen_edges: set[tuple] = set()

    for etype, (rel, src_table, dst_table) in _EDGE_META.items():
        src_type = _TABLE_TO_TYPE[src_table]
        dst_type = _TABLE_TO_TYPE[dst_table]
        for sid in list(nodes.keys()):
            for cypher, the_id in [
                (f"MATCH (a:{src_table} {{id: $id}})-[r:{rel}]->(b:{dst_table}) "
                 f"RETURN a.id, a.label, b.id, b.label, r.confidence", sid),
                (f"MATCH (a:{src_table})-[r:{rel}]->(b:{dst_table} {{id: $id}}) "
                 f"RETURN a.id, a.label, b.id, b.label, r.confidence", sid),
            ]:
                try:
                    res = conn.execute(cypher, {"id": the_id})
                    while res.has_next():
                        aid, albl, bid, blbl, conf = res.get_next()
                        nodes.setdefault(aid, {"id": aid, "type": src_type, "label": albl, "date": None, "status": "normal"})
                        nodes.setdefault(bid, {"id": bid, "type": dst_type, "label": blbl, "date": None, "status": "normal"})
                        key = (aid, bid, etype)
                        if key not in seen_edges:
                            seen_edges.add(key)
                            edges.append({"from": aid, "to": bid, "type": etype, "confidence": float(conf or 0.5)})
                except Exception as e:
                    log.debug("kuzu edge query (%s): %s", etype, e)

    return {"nodes": list(nodes.values()), "edges": edges, "alerts": []}
