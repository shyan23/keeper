"""Chatbot node: answer health-graph questions + return a subgraph for the UI.

Flow:
  1. LLM extracts entity name from question
  2. Redis full-graph cache → fast path (no DB)
  3. Kuzu subgraph query (in-process, fast)
  4. LLM formats a concise medical answer
  5. Returns {answer, graph_result, messages} — subgraph piggybacks in the message
"""
from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel

from app import cache
from app.services import kuzu_graph as kgraph

log = logging.getLogger(__name__)

_GRAPH_CACHE_KEY = "keeper:graph:{patient_id}"
_SUBGRAPH_CACHE_KEY = "keeper:subgraph:{patient_id}:{entity}"
_SUBGRAPH_TTL = 1800  # 30 min


class _EntityExtract(BaseModel):
    entity: str | None = None  # entity name, or null for full-graph question


_EXTRACT_PROMPT = """\
Extract the medical entity the user is asking about (a disease, medication, or test name).
Return JSON: {{"entity": "<name or null if asking about the whole health graph>"}}

Question: {question}
JSON:"""

_ANSWER_PROMPT = """\
You are a medical AI. The user asked: {question}

Relevant health graph ({n_nodes} nodes, {n_edges} relationships):
{subgraph}

Answer concisely (max 120 words). Mention specific relationships and any abnormal values.\
"""


def graph_query_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    patient_id: int | None = state.get("patient_id")
    messages: list[dict] = state.get("messages", [])
    question = messages[-1].get("content", "") if messages else ""

    if not patient_id:
        answer = "Please select a patient first so I can show you their health graph."
        return {"answer": answer, "messages": messages + [{"role": "assistant", "content": answer}]}

    entity = _extract_entity(question, deps, config)
    subgraph = _get_subgraph(patient_id, entity, deps)

    if not subgraph.get("nodes"):
        # Subgraph is empty — fall back to full graph from cache
        subgraph = _get_full_graph_cached(patient_id, deps)

    answer = _format_answer(question, subgraph, deps)
    msg = {"role": "assistant", "content": answer, "subgraph": subgraph}
    return {"answer": answer, "graph_result": subgraph, "messages": messages + [msg]}


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_entity(question: str, deps: Any, config: dict) -> str | None:
    try:
        ex = deps.chat.structured(
            _EXTRACT_PROMPT.format(question=question), _EntityExtract, config=config
        )
        return ex.entity
    except Exception:
        return None


def _get_subgraph(patient_id: int, entity: str | None, deps: Any) -> dict:
    if entity:
        cache_key = _SUBGRAPH_CACHE_KEY.format(
            patient_id=patient_id,
            entity=entity.lower().replace(" ", "_")[:40],
        )
        r = cache._client()
        if r:
            try:
                hit = r.get(cache_key)
                if hit:
                    return json.loads(hit)
            except Exception:
                pass
        sg = kgraph.query_subgraph(patient_id, entity)
        if r and sg.get("nodes"):
            try:
                r.set(cache_key, json.dumps(sg, default=str), ex=_SUBGRAPH_TTL)
            except Exception:
                pass
        return sg
    return {"nodes": [], "edges": [], "alerts": []}


def _get_full_graph_cached(patient_id: int, deps: Any) -> dict:
    """Return cached full graph, or rebuild from Postgres if cache miss."""
    cache_key = _GRAPH_CACHE_KEY.format(patient_id=patient_id)
    r = cache._client()
    if r:
        try:
            hit = r.get(cache_key)
            if hit:
                return json.loads(hit)
        except Exception:
            pass
    # Cache miss — rebuild (slow path, only on first ever query)
    try:
        from app.services.graph import build_graph
        with deps.session_factory() as db:
            graph = build_graph(db, patient_id)
        kgraph.ingest(patient_id, graph)
        if r:
            try:
                r.set(cache_key, json.dumps(graph, default=str), ex=3600)
            except Exception:
                pass
        return graph
    except Exception as e:
        log.warning("full graph rebuild failed: %s", e)
        return {"nodes": [], "edges": [], "alerts": []}


def _format_answer(question: str, subgraph: dict, deps: Any) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    node_map = {n["id"]: n["label"] for n in nodes}
    summary = {
        "nodes": [{"label": n["label"], "type": n["type"], "status": n.get("status")} for n in nodes[:20]],
        "relationships": [
            {
                "from": node_map.get(e["from"], e["from"]),
                "relation": e["type"].replace("_", " "),
                "to": node_map.get(e["to"], e["to"]),
                "confidence": round(e["confidence"], 2),
            }
            for e in edges[:30]
        ],
    }
    prompt = _ANSWER_PROMPT.format(
        question=question,
        n_nodes=len(nodes),
        n_edges=len(edges),
        subgraph=json.dumps(summary, indent=2),
    )
    try:
        return deps.chat.complete(prompt)
    except Exception as e:
        log.warning("graph answer LLM failed: %s", e)
        if nodes:
            labels = ", ".join(n["label"] for n in nodes[:5])
            return f"Found {len(nodes)} related entities: {labels}."
        return "No matching entities found in the health graph."
