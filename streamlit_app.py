"""Streamlit surface for the medical-document tracker.

Two views: a dashboard over the tested service layer, and an agent Chat that
drives the LangGraph supervisor (ingest / structured-query / RAG) with
human-in-the-loop approval gates. Run: `streamlit run streamlit_app.py`.
"""
import uuid
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from app.db import SessionLocal
from app.services import browse as bsvc
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc

st.set_page_config(page_title="Medical Document Tracker", layout="wide")


def _ext(filename: str) -> str:
    suffix = Path(filename).suffix.lstrip(".").lower()
    return suffix or "bin"


# --------------------------------------------------------------------------- #
# Browse views (read-only organization pages). Upload lives in Chat.
# --------------------------------------------------------------------------- #

def _scope_caption(active_pid, selected_label) -> None:
    if active_pid is not None:
        st.caption(f"Showing **{selected_label}** only — change in the sidebar.")
    else:
        st.caption("Showing **all patients** — pick one in the sidebar to filter.")


def patients_page(db, patients, active_pid) -> None:
    st.title("🧑 Patients")
    c1, c2 = st.columns(2)
    c1.metric("Patients", len(patients))
    c2.metric("Documents", dsvc.count_documents(db, patient_id=active_pid))
    if patients:
        st.dataframe(
            [
                {"id": p.id, "name": p.name, "age": p.age, "gender": p.gender,
                 "relationship": p.relationship,
                 "documents": dsvc.count_documents(db, patient_id=p.id)}
                for p in patients
            ],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No patients yet. Add one in the sidebar, then upload a document in **Chat**.")


def _provenance_expander(rows, label_key, source_lines) -> None:
    """A real 'source' control: collapsed by default, shows the document span
    that proves each row (the messy OCR text lives here, out of the main table)."""
    with st.expander("🔎 Sources — the document span that proves each row"):
        for r in source_lines:
            span = (r.get("source") or "").strip()
            if span:
                st.markdown(f"**{r[label_key]}** · doc #{r.get('document_id')}  \n"
                            f"<small>{span}</small>", unsafe_allow_html=True)


def _entity_page(db, title, entity_type, active_pid, selected_label, empty_msg) -> None:
    st.title(title)
    _scope_caption(active_pid, selected_label)
    rows = bsvc.list_entity_links(db, entity_type, patient_id=active_pid)
    if not rows:
        st.info(empty_msg)
        return
    cols = ["name", "confidence", "doc_type", "date"]
    if active_pid is None:
        cols.insert(1, "patient")
    st.dataframe([{c: r.get(c) for c in cols} for r in rows],
                 use_container_width=True, hide_index=True)
    st.caption(f"{len(rows)} record(s).")
    _provenance_expander(rows, "name", rows)


def diagnostics_page(db, active_pid, selected_label) -> None:
    st.title("🧪 Diagnostics")
    _scope_caption(active_pid, selected_label)
    rows = bsvc.list_test_results(db, patient_id=active_pid)
    if not rows:
        st.info("No test results yet. Upload a lab report in **Chat**.")
        return
    cols = ["test", "value", "unit", "reference_range", "date"]
    if active_pid is None:
        cols.insert(1, "patient")
    st.dataframe([{c: r.get(c) for c in cols} for r in rows],
                 use_container_width=True, hide_index=True)
    st.caption(f"{len(rows)} test result(s).")
    _provenance_expander(rows, "test", rows)


def documents_page(db, active_pid, selected_label) -> None:
    st.title("📅 Documents")
    _scope_caption(active_pid, selected_label)
    rows = bsvc.list_documents_timeline(db, patient_id=active_pid)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No documents yet. Upload one in **Chat**.")


# --------------------------------------------------------------------------- #
# Chat view (agent graph + HITL)
# --------------------------------------------------------------------------- #

@st.cache_resource
def _graph():
    from app.agent.graph import build_graph
    return build_graph()


@st.cache_resource
def _deps():
    # build_deps probes the embedder once (network); cached as a singleton.
    from app.agent.providers import build_deps
    return build_deps(SessionLocal)


def _cfg():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    return {"configurable": {"deps": _deps(), "thread_id": st.session_state.thread_id}}


_NODE_LABELS = {
    "router": "🧭 Routing your request…",
    "extract_text": "📖 Reading the document (OCR)…",
    "extract_entities": "🔬 Extracting patient, symptoms, meds, tests…",
    "resolve_patient": "🧑 Matching patient…",
    "persist": "💾 Saving entities…",
    "chunk_embed": "📚 Indexing for search…",
    "parse_filters": "🔎 Parsing your query…",
    "query_db": "🗂️ Looking up records…",
    "transform_query": "✍️ Reformulating the query…",
    "retrieve": "🔍 Searching documents…",
    "rerank": "📊 Ranking results…",
    "grade": "⚖️ Checking answer confidence…",
    "correct_query": "🔁 Refining the search…",
    "generate_answer": "🧠 Composing the answer…",
}


def _drive(graph, cfg, payload) -> None:
    """Stream (or resume) the graph with live progress; capture an interrupt or refresh log."""
    interrupt_val = None
    try:
        with st.status("Working…", expanded=True) as status:
            # Sub-line updated live by long nodes (per-page OCR) via the progress
            # callback; Streamlit flushes placeholder writes mid-node (same thread).
            detail = st.empty()
            cfg["configurable"]["progress"] = lambda msg: detail.write(f"   ↳ {msg}")
            for chunk in graph.stream(payload, cfg, stream_mode="updates"):
                for node in chunk:
                    if node == "__interrupt__":
                        interrupt_val = chunk["__interrupt__"][0].value
                        continue
                    detail.empty()
                    status.write(_NODE_LABELS.get(node, f"… {node}"))
            status.update(
                label="⏸️ Paused for your approval" if interrupt_val else "✅ Done",
                state="complete",
            )
    except Exception as e:  # noqa: BLE001 - surface provider/OCR failures instead of a blank screen
        st.error(f"The agent hit an error: {e}")
        return

    if interrupt_val is not None:
        st.session_state.pending_interrupt = interrupt_val
    else:
        st.session_state.pending_interrupt = None
        snap = graph.get_state(cfg)
        st.session_state.chat_log = snap.values.get("messages", st.session_state.chat_log)
    st.rerun()


def _summarize_extracted(ex: dict) -> None:
    """Compact, human-readable view of an ExtractionResult dict (no raw JSON)."""
    meta = " · ".join(x for x in [ex.get("doc_type"), ex.get("doc_date"),
                                  ex.get("doctor")] if x)
    if meta:
        st.caption(meta)
    tests = ex.get("tests") or []
    if tests:
        st.markdown(f"**Test results ({len(tests)})**")
        st.dataframe(
            [{"test": t.get("name"), "value": t.get("value"), "unit": t.get("unit"),
              "reference": t.get("reference_range")} for t in tests],
            use_container_width=True, hide_index=True,
        )
    for key, label in [("diseases", "Diseases"), ("symptoms", "Symptoms"),
                       ("medications", "Medications")]:
        items = ex.get(key) or []
        if items:
            st.markdown(f"**{label}:** " + ", ".join(
                i.get("name", "") for i in items if i.get("name")))


def _render_interrupt(graph, cfg, payload) -> None:
    kind = payload.get("type")
    st.warning(f"⏸️ Needs your approval — {kind}")

    if kind == "confirm_ingest":
        ex = payload.get("extracted") or {}
        pre_id = payload.get("patient_id")
        cands = payload.get("candidates", [])

        # --- Patient (single section) ---
        st.subheader("🧑 Patient")
        if pre_id:
            st.success(f"Matched existing patient **{ex.get('patient_name') or ''}** (#{pre_id}). "
                       "Saving under this profile.")
            existing = str(pre_id)
        else:
            if cands:
                st.write("Possible existing matches:",
                         ", ".join(f"{c['name']} (#{c['id']})" for c in cands))
                existing = st.text_input("Use an existing patient id (blank = create new)")
            else:
                st.caption("New patient — a profile will be created from the fields below.")
                existing = ""
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Name", value=ex.get("patient_name") or "")
            age = c2.number_input("Age", min_value=0, max_value=130,
                                  value=int(ex.get("patient_age") or 0))
            gender = c3.text_input("Gender", value=ex.get("patient_gender") or "")

        # --- Extracted data (compact summary, not raw JSON) ---
        st.subheader("📋 Extracted data")
        _summarize_extracted(ex)
        with st.expander("Edit raw JSON (advanced)"):
            import json
            edited = st.text_area("Entities (JSON)",
                                  value=json.dumps(ex, indent=2, default=str), height=240)

        a1, a2 = st.columns(2)
        if a1.button("✅ Approve & save", type="primary"):
            import json
            try:
                data = json.loads(edited)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return
            resume = {"approved": True, "extracted": data}
            if pre_id:
                resume["patient_id"] = int(pre_id)
            elif existing.strip():
                resume["patient_id"] = int(existing)
            else:
                resume.update({"name": name.strip() or None,
                               "age": int(age) or None,
                               "gender": gender.strip() or None})
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume=resume))
        if a2.button("❌ Reject"):
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume={"approved": False}))

    elif kind == "low_confidence":
        st.write(f"Weak retrieval (score {payload.get('score')}). Answer anyway?")
        with st.expander("Retrieved snippets"):
            st.json(payload.get("snippets", []))
        c1, c2 = st.columns(2)
        if c1.button("Answer anyway"):
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume={"proceed": True}))
        if c2.button("Skip"):
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume={"proceed": False}))


def chat_page(db, patients, label_to_id, active_pid) -> None:
    st.title("💬 Agent Chat")
    st.caption("Upload a document and say *“read this and arrange it”*, ask for the "
               "*latest report of <patient>*, or ask a question about the records.")

    st.session_state.setdefault("chat_log", [])

    try:
        graph = _graph()
        cfg = _cfg()
    except Exception as e:  # noqa: BLE001 - surface provider/setup failures to the user
        st.error(f"Agent unavailable: {e}\n\nStart Ollama (or set a valid Gemini/Groq key) and reload.")
        return

    # An open HITL gate takes over the page until resolved.
    pending = st.session_state.get("pending_interrupt")
    if pending:
        _render_interrupt(graph, cfg, pending)
        return

    for m in st.session_state.chat_log:
        st.chat_message(m["role"]).write(m["content"])

    up = st.file_uploader("Attach a document — I start reading it automatically",
                          type=["png", "jpg", "jpeg", "pdf", "webp", "txt"])
    prompt = st.chat_input("Ask a question, or attach a document above")

    # Auto-start: the moment a NEW file is attached, kick off ingest without making
    # the user type. The agent OCRs, extracts the name, and arranges it (HITL gates
    # still pause for verification). Re-runs skip an already-processed file.
    if up is not None:
        sig = (up.name, up.size)
        if st.session_state.get("last_upload_sig") != sig:
            st.session_state.last_upload_sig = sig
            import app.storage as storage
            ext = _ext(up.name)
            mime = up.type or ("application/pdf" if ext == "pdf"
                               else "text/plain" if ext == "txt" else f"image/{ext}")
            staged = storage.save_staging(ext, up.getvalue())
            state = {
                "messages": st.session_state.chat_log + [
                    {"role": "user", "content": prompt or "Read this and arrange it."}],
                "file_path": staged, "mime_type": mime, "file_ext": ext,
                "source_type": "pdf" if ext == "pdf" else "image",
            }
            _drive(graph, cfg, state)
            return

    if not prompt:
        return

    # Text-only question: scope to the active patient.
    state = {"messages": st.session_state.chat_log + [{"role": "user", "content": prompt}]}
    if active_pid is not None:
        state["patient_id"] = active_pid
    else:
        st.warning("Pick an active patient in the sidebar to ask about their records, "
                   "or attach a document for me to read and arrange.")
        return

    _drive(graph, cfg, state)


# --------------------------------------------------------------------------- #
# Shared sidebar + view router
# --------------------------------------------------------------------------- #

def main() -> None:
    db = SessionLocal()
    try:
        patients = psvc.list_patients(db)
        label_to_id = {f"{p.name} (#{p.id})": p.id for p in patients}

        with st.sidebar:
            view = st.radio(
                "View",
                ["Chat", "Patients", "Diseases", "Symptoms", "Medications",
                 "Diagnostics", "Documents"],
            )
            health = hsvc.check_health()
            if health["db"] == "ok" and health["pgvector"]:
                st.caption(f"DB ok · pgvector ready · v{health['version']}")
            else:
                st.error(f"Backend not ready: {health}")

            st.header("Patients")
            selected_label = st.selectbox("Active patient", ["— all —"] + list(label_to_id))
            active_pid = label_to_id.get(selected_label)

            with st.form("add_patient", clear_on_submit=True):
                st.subheader("Add patient")
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=0, max_value=130, value=0, step=1)
                gender = st.selectbox("Gender", ["", "male", "female", "other"])
                relationship = st.text_input("Relationship (e.g. mother, self)")
                if st.form_submit_button("Save patient") and name.strip():
                    psvc.create_patient(
                        db, name=name.strip(), age=int(age) or None,
                        gender=gender or None, relationship=relationship.strip() or None,
                    )
                    st.success(f"Added {name}")
                    st.rerun()

        if view == "Chat":
            chat_page(db, patients, label_to_id, active_pid)
        elif view == "Patients":
            patients_page(db, patients, active_pid)
        elif view == "Diseases":
            _entity_page(db, "🦠 Diseases", "disease", active_pid, selected_label,
                         "No diseases extracted yet. Upload a document in Chat.")
        elif view == "Symptoms":
            _entity_page(db, "🤒 Symptoms", "symptom", active_pid, selected_label,
                         "No symptoms extracted yet. Upload a document in Chat.")
        elif view == "Medications":
            _entity_page(db, "💊 Medications", "medication", active_pid, selected_label,
                         "No medications extracted yet. Upload a document in Chat.")
        elif view == "Diagnostics":
            diagnostics_page(db, active_pid, selected_label)
        elif view == "Documents":
            documents_page(db, active_pid, selected_label)
    finally:
        db.close()


main()
