from __future__ import annotations

from typing import Any

from langgraph.types import interrupt
from pydantic import BaseModel

from app.services.edits import apply_edit, find_edit_target

_EDIT_PROMPT = """The user wants to correct ONE extracted medical record. Extract:
- field: one of test_value, test_unit, test_reference, disease, symptom, medication, doc_type, report_date
  (use test_value for a test's numeric/text result, e.g. "set hemoglobin to 1.2")
- target_name: the name of the test/disease/symptom/medication being edited (e.g. "hemoglobin"); null for doc_type/report_date
- new_value: the corrected value/name/date the user wants
- which_document: "latest" unless the user names a specific document or date
- doc_type: a document type hint if mentioned (e.g. "lab report"), else null
Request: {text}"""


class EditPlan(BaseModel):
    field: str = ""
    target_name: str | None = None
    new_value: str | None = None
    which_document: str = "latest"
    doc_type: str | None = None


def _last_user_text(state: dict[str, Any]) -> str:
    for m in reversed(state.get("messages", [])):
        if m.get("role") == "user":
            return m.get("content", "")
    return ""


def _say(state: dict[str, Any], msg: str, **extra: Any) -> dict[str, Any]:
    return {"answer": msg,
            "messages": state["messages"] + [{"role": "assistant", "content": msg}],
            **extra}


def plan_edit_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Parse the edit request and locate the target record. Produces a proposal for
    the human to verify (no DB write happens here)."""
    deps = config["configurable"]["deps"]
    pid = state.get("patient_id")
    if not pid:
        return _say(state, "Select a patient first, then tell me what to edit.",
                    edit_target=None)
    plan = deps.chat.structured(_EDIT_PROMPT.format(text=_last_user_text(state)),
                                EditPlan, config=config).model_dump()
    with deps.session_factory() as s:
        target = find_edit_target(s, int(pid), plan)
    if not target:
        what = plan.get("target_name") or plan.get("field") or "that record"
        return _say(state, f"I couldn't find “{what}” in this patient's records to edit.",
                    edit_target=None)
    return {"edit_target": target}


def confirm_edit_node(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """HITL verify gate: show current -> proposed, let the human edit the value or
    cancel. Only on approval is the change written to the database."""
    deps = config["configurable"]["deps"]
    target = state.get("edit_target")
    if not target:
        return {}
    decision = interrupt({"type": "confirm_edit", "edit": target})
    if not decision.get("approved"):
        return _say(state, "Edit cancelled — nothing changed.")
    proposed = decision.get("proposed", target.get("proposed"))
    target = {**target, "proposed": proposed}
    with deps.session_factory() as s:
        apply_edit(s, target)
    return _say(state, f"Updated {target['label']} to “{proposed}” "
                       f"({target.get('doc_type')}, {target.get('date') or 'undated'}).")
