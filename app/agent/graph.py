from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.router import classify_intent
from app.agent.state import AgentState
from app.agent.nodes.ingest import (
    chunk_embed_node, confirm_ingest_node, create_document_node, dedup_check_node,
    extract_entities_node, extract_text_node, persist_node, resolve_patient_node,
)
from app.agent.nodes.structured import parse_filters_node, query_db_node
from app.agent.nodes.rag import (
    confirm_low_confidence_node, correct_query_node, generate_answer_node,
    grade_node, rerank_node, retrieve_node, transform_query_node,
)


def _route(state: AgentState) -> str:
    return state.get("intent") or "rag_query"


def _after_confirm_ingest(state: AgentState) -> str:
    return "rejected" if state.get("intent") == "rejected" else "create_document"


def _crag_route(state: AgentState) -> str:
    if state.get("low_confidence") and not state.get("corrected"):
        return "correct"
    return "proceed"


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)

    g.add_node("router", classify_intent)
    # ingest
    g.add_node("dedup_check", dedup_check_node)
    g.add_node("extract_text", extract_text_node)
    g.add_node("extract_entities", extract_entities_node)
    g.add_node("resolve_patient", resolve_patient_node)
    g.add_node("confirm_ingest", confirm_ingest_node)
    g.add_node("create_document", create_document_node)
    g.add_node("persist", persist_node)
    g.add_node("chunk_embed", chunk_embed_node)
    # structured
    g.add_node("parse_filters", parse_filters_node)
    g.add_node("query_db", query_db_node)
    # rag
    g.add_node("transform_query", transform_query_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("grade", grade_node)
    g.add_node("correct_query", correct_query_node)
    g.add_node("confirm_low_confidence", confirm_low_confidence_node)
    g.add_node("generate_answer", generate_answer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route, {
        "ingest": "dedup_check",
        "structured_query": "parse_filters",
        "rag_query": "transform_query",
    })

    # ingest chain — single approval gate (patient + entities reviewed together)
    g.add_conditional_edges("dedup_check", lambda s: s.get("dedup", "new"),
                            {"duplicate": END, "new": "extract_text"})
    g.add_edge("extract_text", "extract_entities")
    g.add_edge("extract_entities", "resolve_patient")
    g.add_edge("resolve_patient", "confirm_ingest")
    g.add_conditional_edges("confirm_ingest", _after_confirm_ingest, {
        "rejected": END, "create_document": "create_document",
    })
    g.add_edge("create_document", "persist")
    g.add_edge("persist", "chunk_embed")
    g.add_edge("chunk_embed", END)

    # structured chain
    g.add_edge("parse_filters", "query_db")
    g.add_edge("query_db", END)

    # rag chain (HyDE -> retrieve -> rerank -> grade -> [CRAG correct loop] -> HITL -> answer)
    g.add_edge("transform_query", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges("grade", _crag_route, {
        "correct": "correct_query",
        "proceed": "confirm_low_confidence",
    })
    g.add_edge("correct_query", "retrieve")
    g.add_edge("confirm_low_confidence", "generate_answer")
    g.add_edge("generate_answer", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())
