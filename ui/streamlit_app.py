"""
CiteMind — Streamlit UI (Phase A).

This is a thin client: it only calls the FastAPI backend over HTTP.
It does NOT import any backend/RAG code directly. This means when we
later replace this file with a React app, the backend needs zero changes.
"""

import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="CiteMind", layout="wide")
st.title("CiteMind")
st.caption("Ask My Docs — with enforced citations and conflict detection.")

# --- Session state ---
if "notebook_id" not in st.session_state:
    st.session_state.notebook_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar: create/select notebook ---
with st.sidebar:
    st.header("Notebook")
    notebook_name = st.text_input("New notebook name", value="My Notebook")
    if st.button("Create notebook"):
        resp = requests.post(f"{API_URL}/notebooks", params={"name": notebook_name})
        if resp.ok:
            st.session_state.notebook_id = resp.json()["notebook_id"]
            st.success(f"Created notebook: {notebook_name}")
        else:
            st.error("Failed to create notebook — is the backend running?")

    st.divider()
    st.header("Upload document")
    uploaded_file = st.file_uploader("Choose a PDF or CSV", type=["pdf", "csv"])
    if uploaded_file and st.session_state.notebook_id:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        resp = requests.post(
            f"{API_URL}/upload",
            params={"notebook_id": st.session_state.notebook_id},
            files=files,
        )
        if resp.ok:
            st.success(f"Uploaded: {uploaded_file.name}")
        else:
            st.error("Upload failed.")
    elif uploaded_file and not st.session_state.notebook_id:
        st.warning("Create a notebook first.")

# --- Main: chat interface ---
if not st.session_state.notebook_id:
    st.info("Create a notebook in the sidebar to get started.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
            with st.expander("Citations"):
                for c in entry["citations"]:
                    st.markdown(f"**{c['doc']}**, page {c['page']}: _{c['snippet']}_")

    question = st.chat_input("Ask a question about your documents...")
    if question:
        resp = requests.post(
            f"{API_URL}/query",
            json={"notebook_id": st.session_state.notebook_id, "question": question},
        )
        if resp.ok:
            data = resp.json()
            st.session_state.chat_history.append(
                {"question": question, "answer": data["answer"], "citations": data["citations"]}
            )
            st.rerun()
        else:
            st.error("Query failed — is the backend running?")
