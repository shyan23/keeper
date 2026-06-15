from __future__ import annotations

from typing import Any

_VALID = {"ingest", "structured_query", "rag_query"}

_PROMPT = """Classify the user's request into exactly one label:
- structured_query: asking for a specific document/record by patient, type, or recency (e.g. "latest report of Jane", "show prescriptions for Bob").
- rag_query: a question about the CONTENT of documents (e.g. "what did the doctor say about her blood pressure?").
Respond with ONLY the label.

User: {text}
Label:"""


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def classify_intent(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    # A pending file upload always means ingest.
    if state.get("file_path"):
        return {"intent": "ingest"}
    deps = config["configurable"]["deps"]
    label = deps.chat.complete(_PROMPT.format(text=_last_user_text(state))).strip().lower()
    if "structured" in label:
        return {"intent": "structured_query"}
    if "ingest" in label:
        return {"intent": "ingest"}
    return {"intent": "rag_query"}  # safe default
