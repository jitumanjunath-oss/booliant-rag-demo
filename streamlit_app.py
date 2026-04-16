import streamlit as st
import requests
import json

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Booliant RAG Demo", layout="wide")

st.title("Booliant AI Document Intelligence Demo")
st.markdown(
    """
Ask questions against enterprise documents using retrieval-augmented generation (RAG).

This prototype demonstrates:
- document ingestion
- semantic retrieval
- AI-generated answers
- source-backed citations
"""
)

st.warning(
    "This is a demonstration system. Please do not upload sensitive, confidential, "
    "or regulated data. Documents are processed for demo purposes only."
)

if "file_id" not in st.session_state:
    st.session_state.file_id = None
if "saved_as" not in st.session_state:
    st.session_state.saved_as = None
if "chunks_path" not in st.session_state:
    st.session_state.chunks_path = None

tab1, tab2 = st.tabs(["Build Index", "Ask Questions"])

with tab1:
    st.subheader("1. Upload and Process Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"]
    )

    st.caption("Do not upload sensitive or confidential files. PDF recommended.")
    
    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Upload + Build Index"):
            with st.spinner("Uploading file..."):
                files = {
                    "files": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "application/octet-stream")
                }
                ingest_resp = requests.post(f"{API_BASE}/ingest", files=files)

            if ingest_resp.status_code != 200:
                st.error(f"Ingest failed: {ingest_resp.text}")
            else:
                ingest_data = ingest_resp.json()
                saved_as = ingest_data["files"][0]["saved_as"]
                st.session_state.saved_as = saved_as
                st.success("File uploaded successfully.")

                with st.spinner("Building chunks..."):
                    chunks_payload = {
                        "saved_as": saved_as,
                        "chunk_size": 1200,
                        "overlap": 200
                    }
                    chunks_resp = requests.post(f"{API_BASE}/build_chunks", json=chunks_payload)

                if chunks_resp.status_code != 200:
                    st.error(f"Chunking failed: {chunks_resp.text}")
                else:
                    chunks_data = chunks_resp.json()
                    chunks_path = chunks_data["chunks_path"]
                    st.session_state.chunks_path = chunks_path
                    st.info(f"Created {chunks_data['num_chunks']} chunks.")

                    with st.spinner("Building vector index..."):
                        index_payload = {
                            "chunks_path": chunks_path,
                            "embedding_model": "text-embedding-3-small"
                        }
                        index_resp = requests.post(f"{API_BASE}/build_index", json=index_payload)

                    if index_resp.status_code != 200:
                        st.error(f"Indexing failed: {index_resp.text}")
                    else:
                        index_data = index_resp.json()
                        st.session_state.file_id = index_data["file_id"]
                        st.success("Index built successfully.")
                        st.json(index_data)

    if st.session_state.file_id:
        st.write("### Current indexed file")
        st.code(st.session_state.file_id)

with tab2:
    st.subheader("2. Ask Questions")

    if not st.session_state.file_id:
        st.warning("Please upload and index a document first in the Build Index tab.")
    else:
        question = st.text_input(
            "Ask a question about the document",
            placeholder="What are the key requirements or policies described in this document?"
        )

        top_k = st.slider("Top K chunks", min_value=1, max_value=10, value=5)

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Searching and generating answer..."):
                    ask_payload = {
                        "file_id": st.session_state.file_id,
                        "question": question,
                        "top_k": top_k,
                        "embedding_model": "text-embedding-3-small",
                        "answer_model": "gpt-4.1-mini"
                    }
                    ask_resp = requests.post(f"{API_BASE}/ask", json=ask_payload)

                if ask_resp.status_code != 200:
                    st.error(f"Ask failed: {ask_resp.text}")
                else:
                    ask_data = ask_resp.json()

                    st.write("### Answer")
                    st.write(ask_data["answer"])

                    st.write("### Citations")
                    for i, c in enumerate(ask_data["citations"], start=1):
                        with st.expander(f"Source {i}"):
                            st.write(f"**Source file:** {c['source_file']}")
                            st.write(f"**Chunk index:** {c['chunk_index']}")
                            st.write(f"**Score:** {c['score']:.4f}")
                            st.write("**Snippet:**")
                            st.write(c.get("text", ""))
