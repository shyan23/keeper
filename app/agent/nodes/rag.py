from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from app.config import get_settings
from app.services.retrieval import search_chunks

_GRADE_PROMPT = """Rate 0.0-1.0 how well these snippets can answer the question.
Respond with ONLY a number.
Question: {q}
Snippets:
{snips}"""

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


def retrieve_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    s = get_settings()
    with deps.session_factory() as sess:
        hits = search_chunks(sess, patient_id=state["patient_id"],
                             query=_last_user_text(state), embedder=deps.embedder,
                             k=s.rag_top_k)
    return {"retrieved": hits}


def grade_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        return {"low_confidence": True}
    snips = "\n".join(f"[#{h['chunk_id']}] {h['text']}" for h in hits)
    raw = deps.chat.complete(_GRADE_PROMPT.format(q=_last_user_text(state), snips=snips))
    try:
        score = float(raw.strip().split()[0])
    except (ValueError, IndexError):
        score = 0.0
    threshold = get_settings().rag_confidence_threshold
    return {"low_confidence": score < threshold, "grade_score": score}


def confirm_low_confidence_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL gate: weak retrieval -> ask whether to answer anyway."""
    if not state.get("low_confidence"):
        return {}
    decision = interrupt({
        "type": "low_confidence",
        "score": state.get("grade_score"),
        "snippets": state.get("retrieved", []),
    })
    if not decision.get("proceed"):
        return {"retrieved": []}  # forces a refusal downstream
    return {}


def generate_answer_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    deps = config["configurable"]["deps"]
    hits = state.get("retrieved", [])
    if not hits:
        msg = "I don't have relevant information in this patient's documents to answer that."
        return {"answer": msg,
                "citations": [],
                "messages": state["messages"] + [{"role": "assistant", "content": msg}]}
    snips = "\n".join(f"[#{h['chunk_id']}] {h['text']}" for h in hits)
    body = deps.chat.complete(_ANSWER_PROMPT.format(q=_last_user_text(state), snips=snips))
    cites = "\n".join(
        f"  - #{h['chunk_id']} ({h.get('doc_type') or 'doc'}, {h.get('uploaded_at') or ''}): "
        f"\"{h['text'][:120]}\"" for h in hits
    )
    full = f"{body}\n\nSources:\n{cites}"
    return {"answer": body, "citations": hits,
            "messages": state["messages"] + [{"role": "assistant", "content": full}]}
