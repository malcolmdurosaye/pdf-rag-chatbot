import streamlit as st
import cohere
import json
import os
import time
import sqlite3
import numpy as np
from src.pdf_processing import extract_text_from_pdf, split_text
from src.rag_pipeline import create_embeddings, build_faiss_index, retrieve_chunks, generate_answer
from src.evaluation import generate_ground_truth, save_ground_truth, evaluate_retrieval
from src.fine_tuning import generate_fine_tune_dataset, start_fine_tuning
from src.storage import init_db, save_chunks_and_embeddings, load_chunks_and_embeddings, load_chunks_for_file, get_file_names

# Set Cohere API key (from secrets, not UI)
COHERE_API_KEY = st.secrets.get("COHERE_API_KEY", "sS5dzpbakUlfsYsIdLMwFUic1MV8JJZbsSpG89Aa")
co = cohere.Client(COHERE_API_KEY)

# Sidebar: File Upload & Model Selection
st.sidebar.title("PDF RAG Chatbot")
st.sidebar.markdown("Upload multiple PDFs and interact with them using RAG and Cohere.")

DB_PATH = "data/rag.db"
GROUND_TRUTH_PATH = "data/ground_truth.json"
FINE_TUNE_DATASET_PATH = "data/fine_tune_dataset.jsonl"
conn = init_db(DB_PATH)

st.sidebar.subheader("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Choose PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    st.session_state['uploaded_files'] = [f.name for f in uploaded_files]
    all_chunks = []
    all_embeddings_list = []
    all_file_names = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name.replace('.pdf', '')
        with st.spinner(f"Processing {uploaded_file.name}..."):
            text = extract_text_from_pdf(uploaded_file)
            if not text:
                st.error(f"Failed to extract text from {uploaded_file.name}.")
                continue
            chunks = split_text(text)
            embeddings = create_embeddings(chunks, co)
            if embeddings.size == 0:
                st.error(f"Failed to create embeddings for {uploaded_file.name}.")
                continue
            all_chunks.extend(chunks)
            all_embeddings_list.append(embeddings)
            all_file_names.extend([file_name] * len(chunks))
            save_chunks_and_embeddings(conn, file_name, chunks, embeddings)
            st.success(f"{uploaded_file.name} processed ({len(chunks)} chunks).")
    if all_embeddings_list:
        st.session_state['all_embeddings'] = np.vstack(all_embeddings_list)
        st.session_state['all_chunks'] = all_chunks
        st.session_state['all_file_names'] = all_file_names
        st.session_state['index'] = build_faiss_index(st.session_state['all_embeddings'])
        st.sidebar.success(f"Processed {len(st.session_state['uploaded_files'])} files. Total chunks: {len(all_chunks)}")

if not st.session_state.get('uploaded_files'):
    chunks, embeddings, file_names = load_chunks_and_embeddings(conn)
    if chunks:
        st.session_state['all_chunks'] = chunks
        st.session_state['all_embeddings'] = embeddings
        st.session_state['all_file_names'] = file_names
        st.session_state['index'] = build_faiss_index(embeddings)
        st.sidebar.info(f"Loaded {len(chunks)} chunks from {len(set(file_names))} files.")

# Sidebar: Model Selection
st.sidebar.subheader("Model Selection")
use_finetuned_model = st.sidebar.checkbox("Use Fine-Tuned Model")
fine_tuned_model_id = ""
if use_finetuned_model:
    fine_tuned_model_id = st.sidebar.text_input("Fine-Tuned Model ID:", value=st.session_state.get('fine_tuned_model_id', ""))
    if fine_tuned_model_id:
        st.session_state['fine_tuned_model_id'] = fine_tuned_model_id
else:
    st.session_state['fine_tuned_model_id'] = ""

# Sidebar: Storage Management
st.sidebar.subheader("Storage Management")
# Delete specific PDFs
existing_files = get_file_names(conn)
if existing_files:
    files_to_delete = st.sidebar.multiselect("Delete specific PDFs", options=existing_files)
    if st.sidebar.button("Delete Selected PDFs") and files_to_delete:
        cursor = conn.cursor()
        for fname in files_to_delete:
            cursor.execute('DELETE FROM embeddings WHERE file_name = ?', (fname,))
        conn.commit()
        # Update session state
        chunks, embeddings, file_names = load_chunks_and_embeddings(conn)
        st.session_state['all_chunks'] = chunks
        st.session_state['all_embeddings'] = embeddings
        st.session_state['all_file_names'] = file_names
        st.session_state['index'] = build_faiss_index(embeddings)
        st.sidebar.success(f"Deleted: {', '.join(files_to_delete)}")
# Confirmation for clearing all PDFs
clear_all_confirm = st.sidebar.checkbox("Do you want to clear all PDFs in the database?")
if st.sidebar.button("Clear All PDF Storage") and clear_all_confirm:
    cursor = conn.cursor()
    cursor.execute('DELETE FROM embeddings')
    conn.commit()
    st.session_state['uploaded_files'] = []
    st.session_state['all_chunks'] = []
    st.session_state['all_embeddings'] = np.array([])
    st.session_state['all_file_names'] = []
    st.session_state['index'] = None
    st.sidebar.success("All PDF storage cleared.")

# Main: Workflow Steps
st.title("PDF RAG Chatbot")
st.markdown("""
Interact with your PDFs using Retrieval-Augmented Generation and Cohere. We have implemented the following steps below:
1. **Upload PDFs** in the sidebar.
2. **Generate and Edit Ground Truth** for evaluation.
3. **Evaluate Retrieval**.
4. **Generate Fine-Tuning Dataset** and **Start Fine-Tuning**.
5. **Chat with PDFs** using the best model.
""")

# Step 3: Chat Interface
st.header("Chat with PDFs")

# Chatbot UI container
chat_container = st.container()

if st.session_state.get('all_file_names'):
    unique_files = list(set(st.session_state['all_file_names']))
    selected_file = st.selectbox("Select a document to query (or All)", options=["All"] + unique_files, index=0)
else:
    selected_file = "All"

# Display chat history with avatars
with chat_container:
    if 'chat_history' not in st.session_state or not st.session_state['chat_history']:
        st.chat_message("assistant").write("👋 Hi! Ask me anything about your PDFs.")
    else:
        for msg in st.session_state['chat_history']:
            if msg['role'] == "user":
                st.chat_message("user").write(msg['content'])
            else:
                st.chat_message("assistant").write(msg['content'])

# Chat input at the bottom
with st.container():
    user_query = st.chat_input("Type your message here...")
    if user_query and st.session_state.get('all_chunks'):
        time.sleep(1)
        if selected_file != "All":
            file_name = selected_file
            relevant_chunks, _ = retrieve_chunks(
                user_query,
                st.session_state['index'],
                st.session_state['all_chunks'],
                st.session_state['all_embeddings'],
                co,
                file_name=file_name,
                all_file_names=st.session_state['all_file_names']
            )
        else:
            relevant_chunks, _ = retrieve_chunks(
                user_query,
                st.session_state['index'],
                st.session_state['all_chunks'],
                st.session_state['all_embeddings'],
                co
            )
        model = st.session_state.get('fine_tuned_model_id', 'command-r-08-2024')
        answer = generate_answer(user_query, relevant_chunks, co, model)
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        st.session_state['chat_history'].append({"role": "user", "content": user_query})
        st.session_state['chat_history'].append({"role": "bot", "content": answer})
        st.rerun()  # Refresh to show new message

# Footer: Evaluation Criteria & Info
st.markdown("""
---
**Precision@3**: Fraction of top-3 retrieved chunks that are relevant. High values indicate low false positives.  
**MRR (Mean Reciprocal Rank)**: Rank of the first relevant chunk. High values mean relevant chunks are retrieved early.  

""")