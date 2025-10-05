import cohere
import faiss
import numpy as np
import streamlit as st
import time
from typing import List, Tuple, Optional
from requests.exceptions import RequestException
from cohere.errors import TooManyRequestsError
from src.storage import load_chunks_for_file

def create_embeddings(
    chunks: List[str], 
    cohere_client, 
    sleep_time: int = 6
) -> np.ndarray:
    """
    Create embeddings with batching and token limit handling for Trial key.
    Returns stacked embeddings array.
    """
    max_retries = 5
    base_delay = 5
    batch_size = 100
    embeddings = []

    def estimate_tokens(texts: List[str]) -> float:
        # Rough token estimation
        return sum(len(text.split()) for text in texts) * 1.2

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        token_count = estimate_tokens(batch)
        st.info(f"Processing batch {i//batch_size + 1} with {len(batch)} chunks (~{int(token_count)} tokens)")

        for attempt in range(max_retries):
            try:
                response = cohere_client.embed(
                    model='embed-multilingual-v3.0',
                    texts=batch,
                    input_type='search_document'
                )
                batch_embeddings = np.array(response.embeddings, dtype=np.float32)
                embeddings.append(batch_embeddings)
                st.success(f"Batch {i//batch_size + 1} embedded successfully!")
                time.sleep(sleep_time)
                break
            except TooManyRequestsError as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    st.warning(f"Rate limit exceeded on attempt {attempt + 1}: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                st.error(f"Failed to embed batch {i//batch_size + 1} after {max_retries} attempts: {str(e)}")
                return np.array([])
            except RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    st.warning(f"Network error on attempt {attempt + 1}: {str(e)}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                st.error(f"Failed to embed batch {i//batch_size + 1} after {max_retries} attempts: {str(e)}")
                return np.array([])

    if not embeddings:
        st.error("No embeddings generated. Check PDF size or network.")
        return np.array([])

    # Use vstack for clarity
    return np.vstack(embeddings)

def build_faiss_index(embeddings: np.ndarray) -> Optional[faiss.IndexFlatL2]:
    """
    Build FAISS index for similarity search.
    """
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: List[str],
    embeddings: np.ndarray,
    cohere_client,
    k: int = 3,
    file_name: Optional[str] = None,
    all_file_names: Optional[List[str]] = None
) -> Tuple[List[str], List[int]]:
    """
    Retrieve top-k relevant chunks for a query. If file_name is provided, filter chunks/embeddings for that file.
    """
    if file_name and all_file_names:
        # Filter chunks and embeddings for the selected file
        filtered_indices = [i for i, fname in enumerate(all_file_names) if fname == file_name]
        filtered_chunks = [chunks[i] for i in filtered_indices]
        filtered_embeddings = embeddings[filtered_indices] if len(filtered_indices) > 0 else np.array([])
        if filtered_embeddings.size == 0:
            st.warning(f"No chunks found for {file_name}; querying all files.")
            filtered_chunks = chunks
            filtered_embeddings = embeddings
        filtered_index = build_faiss_index(filtered_embeddings)
    else:
        filtered_chunks = chunks
        filtered_embeddings = embeddings
        filtered_index = index
    try:
        query_response = cohere_client.embed(
            model='embed-multilingual-v3.0',
            texts=[query],
            input_type='search_query'
        )
        query_embedding = np.array(query_response.embeddings, dtype=np.float32)
        distances, indices = filtered_index.search(query_embedding, k)
        return [filtered_chunks[i] for i in indices[0]], indices[0].tolist()
    except Exception as e:
        st.error(f"Error retrieving chunks for query '{query}': {str(e)}")
        return [], []

def generate_answer(
    query: str, 
    relevant_chunks: List[str], 
    cohere_client, 
    model: str = 'command-r-08-2024'
) -> str:
    """
    Generate answer using Cohere API and retrieved chunks.
    """
    context = "\n".join(relevant_chunks)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    try:
        response = cohere_client.chat(model=model, message=prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Unable to generate answer."