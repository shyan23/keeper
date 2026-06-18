from __future__ import annotations

from typing import Any

_VALID = {"ingest", "structured_query", "rag_query", "edit", "generate_pdf"}

_PROMPT = """Classify the user's request into exactly one label:
- generate_pdf: asking to MAKE/CREATE/GENERATE a PDF/report/document OUT OF records (e.g. "make a pdf of all lipid results of patient X for the last 3 years", "make a pdf from all reports of Bob").
- edit: asking to CHANGE/CORRECT/FIX/UPDATE/SET an extracted value, name, or date (e.g. "set hemoglobin to 1.2", "correct the report date to 5 Oct 2023", "rename the diagnosis to anemia").
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
    if "generate_pdf" in label or "pdf" in label:
        return {"intent": "generate_pdf"}
    if "edit" in label:
        return {"intent": "edit"}
    if "structured" in label:
        return {"intent": "structured_query"}
    if "ingest" in label:
        return {"intent": "ingest"}
    return {"intent": "rag_query"}  # safe default
