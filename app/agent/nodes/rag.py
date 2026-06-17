from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.config import get_settings
from app.services.retrieval import search_chunks

_HYDE_PROMPT = """Write a brief, plausible answer paragraph to this medical question as if quoting a patient's medical record. Used only to improve document retrieval; do not refuse.
Question: {q}"""

_RERANK_PROMPT = """Rate 0.0-1.0 how relevant this snippet is to the question. Respond with ONLY a number.
Question: {q}
Snippet: {snip}"""

_GRADE_PROMPT = """Rate 0.0-1.0 how well these snippets can answer the question.
Respond with ONLY a number.
Question: {q}
Snippets:
{snips}"""

_CORRECT_PROMPT = """A search for this question returned weak results. Rewrite it into a more specific search query using likely medical terms/synonyms. Output ONLY the rewritten query.
Question: {q}"""

_ANSWER_PROMPT = """Answer the question USING ONLY the snippets. Do not use outside knowledge.
If the snippets don't contain the answer, say you don't have that information.
Question: {q}
Snippets:
{snips}"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _to_score(raw: str) -> float:
    try:
        return float(raw.strip().split()[0])
    except (ValueError, IndexError, AttributeError):
        return 0.0


def transform_query_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HyDE: generate a hypothetical answer to embed instead of the raw question."""
    deps = config["configurable"]["deps"]
    q = _last_user_text(state)
    try:
        hyde = deps.chat.complete(_HYDE_PROMPT.format(q=q))
    except Exception:  # noqa: BLE001 - HyDE is best-effort
        hyde = ""
    return {"retrieval_query": (hyde or q)}


def retrieve_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    s = get_settings()
    query = state.get("retrieval_query") or _last_user_text(state)
    with deps.session_factory() as sess:
        hits = search_chunks(sess, patient_id=state["patient_id"],
                             query=query, embedder=deps.embedder, k=s.rag_top_k * 3)
    return {"retrieved": hits}


def rerank_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """LLM reranking: score each candidate against the real question, keep top-k."""
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        return {"retrieved": []}
    q = _last_user_text(state)
    scored = []
    for h in hits:
        raw = deps.chat.complete(_RERANK_PROMPT.format(q=q, snip=h["text"]))
        scored.append((_to_score(raw), h))
    scored.sort(key=lambda x: x[0], reverse=True)
    k = get_settings().rag_top_k
    return {"retrieved": [h for _, h in scored[:k]]}


def grade_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        return {"low_confidence": True, "grade_score": 0.0}
    snips = "\n".join(f"[#{h['chunk_id']}] {h['text']}" for h in hits)
    score = _to_score(deps.chat.complete(_GRADE_PROMPT.format(q=_last_user_text(state), snips=snips)))
    threshold = get_settings().rag_confidence_threshold
    return {"low_confidence": score < threshold, "grade_score": score}


def correct_query_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """CRAG: weak retrieval -> rewrite the query for one corrective re-retrieval."""
    deps = config["configurable"]["deps"]
    q = _last_user_text(state)
    rewrite = deps.chat.complete(_CORRECT_PROMPT.format(q=q))
    return {"retrieval_query": (rewrite.strip() if rewrite else q) or q, "corrected": True}


def confirm_low_confidence_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate: weak retrieval WITH some hits -> ask whether to answer anyway.
    Empty retrieval needs no human (nothing to confirm) -> proceed to refusal."""
    hits = state.get("retrieved", [])
    if not state.get("low_confidence") or not hits:
        return {}
    decision = interrupt({
        "type": "low_confidence",
        "score": state.get("grade_score"),
        "snippets": hits,
    })
    if not decision.get("proceed"):
        return {"retrieved": []}
    return {}


def _collapse_sources(hits: list[dict]) -> list[dict]:
    """One citation per document — chunks of the same document collapse into a single,
    user-facing reference (document name/type/date), never chunk/vector ids."""
    seen: dict[Any, dict] = {}
    for h in hits:
        did = h.get("document_id")
        if did in seen:
            continue
        seen[did] = {
            "document_id": did,
            "name": h.get("original_name") or h.get("doc_type") or "Document",
            "doc_type": h.get("doc_type") or "Document",
            "date": h.get("report_date") or None,
        }
    return list(seen.values())


def generate_answer_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        msg = "I don't have that information in this patient's records."
        return {"answer": msg, "citations": [], "sources": [],
                "messages": state["messages"] + [{"role": "assistant", "content": msg, "sources": []}]}
    snips = "\n".join(f"[{i + 1}] {h['text']}" for i, h in enumerate(hits))
    body = deps.chat.complete(_ANSWER_PROMPT.format(q=_last_user_text(state), snips=snips))
    sources = _collapse_sources(hits)
    return {"answer": body, "citations": hits, "sources": sources,
            "messages": state["messages"] + [
                {"role": "assistant", "content": body, "sources": sources}]}
