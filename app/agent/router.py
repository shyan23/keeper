from __future__ import annotations

from typing import Any

from app.agent.state import IntentDecision

CLARIFY_BELOW = 0.80
CONFIRM_BELOW = 0.90

_FALLBACK_QUESTION = (
    "I'm not sure what you'd like me to do — should I look something up, "
    "change a value, or make a report?")

_PROMPT = """You are routing a medical-records assistant. Classify the user's
latest request into exactly one intent, given the recent conversation.

Intents:
- generate_pdf: MAKE/CREATE/GENERATE a PDF/report OUT OF stored records.
- edit: CHANGE/CORRECT/FIX/UPDATE/SET an extracted value, name, or date.
- structured_query: ask for a specific document/record by patient, type, or recency.
- rag_query: a question about the CONTENT of documents.
- ingest: read/store a newly provided document.

Return JSON: {{"intent": <one of the above>, "confidence": <0..1>, "question": <a
short clarifying question if and only if you are unsure, else null>}}.
Set confidence below 0.8 only when the request is genuinely ambiguous.

Recent conversation:
{conversation}

JSON:"""


def _recent_conversation(state: dict[str, Any], n: int = 6) -> str:
    msgs = state.get("messages", [])[-n:]
    return "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs)


def _say(state: dict[str, Any], msg: str, **extra: Any) -> dict[str, Any]:
    return {"answer": msg,
            "messages": state["messages"] + [{"role": "assistant", "content": msg}],
            **extra}


def classify_intent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    # A pending file upload always means ingest.
    if state.get("file_path"):
        return {"intent": "ingest", "route_gate": "go"}

    deps = config["configurable"]["deps"]
    decision: IntentDecision = deps.chat.structured(
        _PROMPT.format(conversation=_recent_conversation(state)), IntentDecision)

    if decision.confidence < CLARIFY_BELOW:
        question = decision.question or _FALLBACK_QUESTION
        return _say(state, question, intent="clarify", route_gate="clarify")

    gate = "confirm" if decision.confidence < CONFIRM_BELOW else "go"
    return {"intent": decision.intent, "route_gate": gate}
