from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agent.router import classify_intent, confirm_intent_node, INTENT_ENTRY
from app.agent.state import AgentState
from app.agent.nodes.ingest import (
    confirm_ingest_node, dedup_check_node, extract_text_node, persist_reports_node,
    resolve_patient_node, segment_extract_node,
)
from app.agent.nodes.structured import parse_filters_node, query_db_node
from app.agent.nodes.edit import confirm_edit_node, plan_edit_node
from app.agent.nodes.rag import (
    confirm_low_confidence_node, correct_query_node, generate_answer_node,
    grade_node, require_patient_node, rerank_node, retrieve_node, transform_query_node,
)
from app.agent.nodes.report import (
    build_report_node, confirm_report_node, deliver_report_node, plan_report_node,
)


def _route(state: AgentState) -> str:
    gate = state.get("route_gate") or "go"
    if gate == "clarify":
        return "clarify"
    if gate == "confirm":
        return "confirm"
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
    g.add_node("confirm_intent", confirm_intent_node)
    # ingest
    g.add_node("dedup_check", dedup_check_node)
    g.add_node("extract_text", extract_text_node)
    g.add_node("segment_extract", segment_extract_node)
    g.add_node("resolve_patient", resolve_patient_node)
    g.add_node("confirm_ingest", confirm_ingest_node)
    g.add_node("persist_reports", persist_reports_node)
    # structured
    g.add_node("parse_filters", parse_filters_node)
    g.add_node("query_db", query_db_node)
    # edit (HITL-verified correction of extracted data)
    g.add_node("plan_edit", plan_edit_node)
    g.add_node("confirm_edit", confirm_edit_node)
    # pdf report
    g.add_node("plan_report", plan_report_node)
    g.add_node("confirm_report", confirm_report_node)
    g.add_node("build_report", build_report_node)
    g.add_node("deliver_report", deliver_report_node)
    # rag
    g.add_node("require_patient", require_patient_node)
    g.add_node("transform_query", transform_query_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("rerank", rerank_node)
    g.add_node("grade", grade_node)
    g.add_node("correct_query", correct_query_node)
    g.add_node("confirm_low_confidence", confirm_low_confidence_node)
    g.add_node("generate_answer", generate_answer_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route, {
        "clarify": END,
        "confirm": "confirm_intent",
        **INTENT_ENTRY,
    })
    g.add_conditional_edges("confirm_intent",
                            lambda s: "clarify" if s.get("route_gate") == "clarify"
                            else (s.get("intent") or "rag_query"),
                            {"clarify": END, **INTENT_ENTRY})

    g.add_edge("require_patient", "transform_query")

    # ingest chain — split into reports, single approval gate, then persist each
    g.add_conditional_edges("dedup_check", lambda s: s.get("dedup", "new"),
                            {"duplicate": END, "new": "extract_text"})
    g.add_edge("extract_text", "segment_extract")
    g.add_edge("segment_extract", "resolve_patient")
    g.add_edge("resolve_patient", "confirm_ingest")
    g.add_conditional_edges("confirm_ingest", _after_confirm_ingest, {
        "rejected": END, "create_document": "persist_reports",
    })
    g.add_edge("persist_reports", END)

    # structured chain
    g.add_edge("parse_filters", "query_db")
    g.add_edge("query_db", END)

    # edit chain — plan, then a single HITL verify gate before any DB write
    g.add_conditional_edges("plan_edit",
                            lambda s: "confirm" if s.get("edit_target") else "end",
                            {"confirm": "confirm_edit", "end": END})
    g.add_edge("confirm_edit", END)

    # pdf report chain — plan -> Gate A -> build -> Gate B -> (download | regenerate)
    g.add_conditional_edges("plan_report",
                            lambda s: "confirm" if s.get("report_plan") else "end",
                            {"confirm": "confirm_report", "end": END})
    g.add_conditional_edges("confirm_report",
                            lambda s: s.get("report_decision") or "end",
                            {"build": "build_report", "replan": "plan_report",
                             "end": END})
    g.add_edge("build_report", "deliver_report")
    g.add_conditional_edges("deliver_report",
                            lambda s: "rebuild" if s.get("report_decision") == "rebuild"
                            else "end",
                            {"rebuild": "build_report", "end": END})

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
