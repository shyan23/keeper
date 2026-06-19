from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Callable

from app.agent.checkpointer import get_checkpointer

logger = logging.getLogger(__name__)

# Maps graph node keys to user-facing progress labels (ported from streamlit_app).
NODE_LABELS: dict[str, str] = {
    "router": "🧭 Routing your request…",
    "extract_text": "📖 Reading the document (OCR)…",
    "segment_extract": "🔬 Splitting reports & extracting…",
    "resolve_patient": "🧑 Matching patient…",
    "persist_reports": "💾 Saving & indexing reports…",
    "parse_filters": "🔎 Parsing your query…",
    "query_db": "🗂️ Looking up records…",
    "plan_edit": "✏️ Finding the record to edit…",
    "confirm_edit": "✅ Awaiting your confirmation…",
    "require_patient": "🧑 Identifying the patient…",
    "transform_query": "✍️ Reformulating the query…",
    "retrieve": "🔍 Searching documents…",
    "rerank": "📊 Ranking results…",
    "grade": "⚖️ Checking answer confidence…",
    "correct_query": "🔁 Refining the search…",
    "generate_answer": "🧠 Composing the answer…",
    "plan_report": "🧾 Planning your report…",
    "confirm_report": "✅ Awaiting your approval…",
    "build_report": "📄 Building the PDF…",
    "deliver_report": "📦 Finalizing your report…",
}


@lru_cache(maxsize=1)
def get_graph():
    """Compile the LangGraph supervisor once with a durable Postgres
    checkpointer. Degrades to in-process MemorySaver if the DB is
    unreachable at boot (non-durable, but the service stays up)."""
    from app.agent.graph import build_graph
    try:
        return build_graph(checkpointer=get_checkpointer())
    except Exception:
        logger.warning(
            "Durable checkpointer unavailable; falling back to MemorySaver "
            "(thread state will NOT survive restart).", exc_info=True,
        )
        return build_graph()


@lru_cache(maxsize=1)
def get_deps():
    """Build agent Deps once (probes the embedder over the network)."""
    from app.agent.providers import build_deps
    from app.db import SessionLocal
    return build_deps(SessionLocal)


def cfg(thread_id: str, deps: Any = None,
        progress: Callable[[str], None] | None = None) -> dict:
    configurable: dict[str, Any] = {
        "deps": deps if deps is not None else get_deps(),
        "thread_id": thread_id,
    }
    if progress is not None:
        configurable["progress"] = progress
    return {"configurable": configurable}
