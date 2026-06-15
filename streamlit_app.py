"""Minimal Streamlit dashboard for the medical-document tracker.

Thin view over the tested service layer in app/services/. No AI yet — this is
the dashboard + upload surface. Run: `streamlit run streamlit_app.py`.
"""
from pathlib import Path

import streamlit as st

from app.db import SessionLocal
from app.services import documents as dsvc
from app.services import health as hsvc
from app.services import patients as psvc


def _ext(filename: str) -> str:
    suffix = Path(filename).suffix.lstrip(".").lower()
    return suffix or "bin"


def main() -> None:
    st.set_page_config(page_title="Medical Document Tracker", layout="wide")
    st.title("🩺 Medical Document Tracker")

    db = SessionLocal()
    try:
        # ---- health banner ----
        health = hsvc.check_health()
        if health["db"] == "ok" and health["pgvector"]:
            st.caption(f"DB ok · pgvector ready · v{health['version']}")
        else:
            st.error(f"Backend not ready: {health}")

        patients = psvc.list_patients(db)

        # ---- sidebar: patient picker + add ----
        with st.sidebar:
            st.header("Patients")
            label_to_id = {f"{p.name} (#{p.id})": p.id for p in patients}
            selected_label = st.selectbox(
                "Active patient",
                ["— all —"] + list(label_to_id),
            )
            active_pid = label_to_id.get(selected_label)

            with st.form("add_patient", clear_on_submit=True):
                st.subheader("Add patient")
                name = st.text_input("Name")
                age = st.number_input("Age", min_value=0, max_value=130, value=0, step=1)
                gender = st.selectbox("Gender", ["", "male", "female", "other"])
                relationship = st.text_input("Relationship (e.g. mother, self)")
                if st.form_submit_button("Save patient") and name.strip():
                    psvc.create_patient(
                        db, name=name.strip(),
                        age=int(age) or None,
                        gender=gender or None,
                        relationship=relationship.strip() or None,
                    )
                    st.success(f"Added {name}")
                    st.rerun()

        # ---- metrics ----
        c1, c2 = st.columns(2)
        c1.metric("Patients", len(patients))
        c2.metric("Documents", dsvc.count_documents(db, patient_id=active_pid))

        # ---- patient table ----
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

        # ---- documents table ----
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

        # ---- upload section (always visible) ----
        st.subheader("📤 Upload document")
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
            # create row first to get an id, then write the file, then store path
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
    finally:
        db.close()


main()
