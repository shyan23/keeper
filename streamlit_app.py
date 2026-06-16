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
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc

st.set_page_config(page_title="Medical Document Tracker", layout="wide")


def _ext(filename: str) -> str:
    suffix = Path(filename).suffix.lstrip(".").lower()
    return suffix or "bin"


# --------------------------------------------------------------------------- #
# Dashboard view
# --------------------------------------------------------------------------- #

def dashboard(db, patients, label_to_id, active_pid, selected_label) -> None:
    st.title("🩺 Medical Document Tracker")

    health = hsvc.check_health()
    if health["db"] == "ok" and health["pgvector"]:
        st.caption(f"DB ok · pgvector ready · v{health['version']}")
    else:
        st.error(f"Backend not ready: {health}")

    c1, c2 = st.columns(2)
    c1.metric("Patients", len(patients))
    c2.metric("Documents", dsvc.count_documents(db, patient_id=active_pid))

    st.subheader("Profiles")
    if patients:
        st.dataframe(
            [
                {"id": p.id, "name": p.name, "age": p.age,
                 "gender": p.gender, "relationship": p.relationship,
                 "documents": dsvc.count_documents(db, patient_id=p.id)}
                for p in patients
            ],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No patients yet. Add one in the sidebar.")

    st.subheader("Documents" + (f" — {selected_label}" if active_pid else ""))
    docs = dsvc.list_documents(db, patient_id=active_pid)
    if docs:
        st.dataframe(
            [
                {"id": d.id, "patient_id": d.patient_id, "type": d.doc_type,
                 "status": d.status, "file": d.file_path,
                 "uploaded": d.uploaded_at.strftime("%Y-%m-%d %H:%M") if d.uploaded_at else None}
                for d in docs
            ],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No documents yet.")

    st.subheader("📤 Upload document")
    st.caption("This only **stores** the file. For OCR + entity extraction + arranging, "
               "use the **Chat** view (sidebar) and say “read this and arrange it”.")
    if patients:
        up_label = st.selectbox("For patient", list(label_to_id), key="upload_patient")
        up_pid = label_to_id[up_label]
    else:
        st.caption("No patients yet — add one in the sidebar to attach this upload.")
        up_pid = None
    doc_type = st.selectbox(
        "Type", ["prescription", "lab_report", "diagnostic_report",
                 "discharge_summary", "other"],
    )
    uploaded = st.file_uploader("File (image or PDF)",
                                type=["png", "jpg", "jpeg", "pdf", "webp"])
    if st.button("Save upload", disabled=(uploaded is None or up_pid is None)):
        data = uploaded.getvalue()
        doc = dsvc.create_document(
            db, patient_id=up_pid, doc_type=doc_type,
            source_type="pdf" if _ext(uploaded.name) == "pdf" else "image",
            mime_type=uploaded.type,
        )
        import app.storage as storage
        path = storage.save_bytes(up_pid, doc.id, _ext(uploaded.name), data)
        dsvc.set_file_path(db, doc.id, path)
        st.success(f"Saved {uploaded.name} → {path} (doc #{doc.id})")
        st.rerun()


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
            for chunk in graph.stream(payload, cfg, stream_mode="updates"):
                for node in chunk:
                    if node == "__interrupt__":
                        interrupt_val = chunk["__interrupt__"][0].value
                        continue
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


def _render_interrupt(graph, cfg, payload) -> None:
    kind = payload.get("type")
    st.warning(f"⏸️ Needs your approval — {kind}")

    if kind == "confirm_entities":
        import json
        st.caption("Review / edit the extracted data, then approve to save it.")
        edited = st.text_area("Extracted entities (JSON)",
                              value=json.dumps(payload["extracted"], indent=2, default=str),
                              height=320)
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve & save"):
            try:
                data = json.loads(edited)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                return
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume={"approved": True, "extracted": data}))
        if c2.button("❌ Reject"):
            st.session_state.pending_interrupt = None
            _drive(graph, cfg, Command(resume={"approved": False}))

    elif kind == "confirm_patient":
        st.write(f"Document name: **{payload.get('extracted_name')}**")
        cands = payload.get("candidates", [])
        st.write("Candidates:", cands or "— none —")
        choice = st.text_input("Existing patient id (leave blank to create new)")
        if st.button("Confirm patient"):
            st.session_state.pending_interrupt = None
            if choice.strip():
                _drive(graph, cfg, Command(resume={"patient_id": int(choice)}))
            else:
                _drive(graph, cfg, Command(resume={"create_new": True}))

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

    up = st.file_uploader("Attach a document (optional)",
                          type=["png", "jpg", "jpeg", "pdf", "webp", "txt"])
    prompt = st.chat_input("Ask, or upload + “read this and arrange it”")
    if not prompt:
        return

    state = {"messages": st.session_state.chat_log + [{"role": "user", "content": prompt}]}

    if up is not None:
        if active_pid is None:
            st.error("Pick an active patient in the sidebar before uploading a document.")
            return
        data = up.getvalue()
        doc = dsvc.create_document(
            db, patient_id=active_pid,
            source_type="pdf" if _ext(up.name) == "pdf" else "image",
            mime_type=up.type,
        )
        import app.storage as storage
        path = storage.save_bytes(active_pid, doc.id, _ext(up.name), data)
        dsvc.set_file_path(db, doc.id, path)
        state.update({"file_path": path, "mime_type": up.type, "document_id": doc.id})
    elif active_pid is not None:
        # RAG / structured queries are scoped to the active patient.
        state["patient_id"] = active_pid

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
            view = st.radio("View", ["Dashboard", "Chat"])
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
        else:
            dashboard(db, patients, label_to_id, active_pid, selected_label)
    finally:
        db.close()


main()
