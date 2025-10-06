import cohere
import faiss
import numpy as np
import streamlit as st
import time
import math
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

    total_batches = max(1, math.ceil(len(chunks) / batch_size)) if chunks else 1
    progress_bar = st.progress(0) if chunks else None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                response = cohere_client.embed(
                    model='embed-multilingual-v3.0',
                    texts=batch,
                    input_type='search_document'
                )
                batch_embeddings = np.array(response.embeddings, dtype=np.float32)
                embeddings.append(batch_embeddings)
                time.sleep(sleep_time)
                if progress_bar is not None:
                    completed_batches = (i // batch_size) + 1
                    progress_value = min(100, int((completed_batches / total_batches) * 100))
                    progress_bar.progress(progress_value)
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

    if progress_bar is not None:
        progress_bar.progress(100)
        progress_bar.empty()

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
) -> Tuple[List[str], List[int], List[str]]:
    """
    Retrieve top-k relevant chunks for a query. If file_name is provided, filter chunks/embeddings for that file.
    Returns chunk texts, their global indices, and the originating file names.
    """

    filtered_indices: Optional[List[int]] = None
    filtered_file_names: Optional[List[str]] = None

    if file_name and all_file_names:
        filtered_indices = [i for i, fname in enumerate(all_file_names) if fname == file_name]
        filtered_chunks = [chunks[i] for i in filtered_indices]
        filtered_embeddings = embeddings[filtered_indices] if len(filtered_indices) > 0 else np.array([])
        if filtered_embeddings.size == 0:
            st.warning(f"No chunks found for {file_name}. Ensure the policy was processed and try again.")
            return [], [], []
        filtered_file_names = [all_file_names[i] for i in filtered_indices]
        filtered_index = build_faiss_index(filtered_embeddings)
    else:
        filtered_chunks = chunks
        filtered_embeddings = embeddings
        filtered_index = index
        if all_file_names:
            filtered_indices = list(range(len(all_file_names)))
            filtered_file_names = all_file_names

    try:
        query_response = cohere_client.embed(
            model='embed-multilingual-v3.0',
            texts=[query],
            input_type='search_query'
        )
        query_embedding = np.array(query_response.embeddings, dtype=np.float32)
        distances, indices = filtered_index.search(query_embedding, k)

        retrieved_chunks: List[str] = []
        retrieved_indices: List[int] = []
        retrieved_files: List[str] = []
        for idx in indices[0]:
            retrieved_chunks.append(filtered_chunks[idx])

            if filtered_indices is not None and idx < len(filtered_indices):
                global_idx = filtered_indices[idx]
            else:
                global_idx = idx
            retrieved_indices.append(global_idx)

            if filtered_file_names is not None and idx < len(filtered_file_names):
                retrieved_files.append(filtered_file_names[idx])
            elif file_name:
                retrieved_files.append(file_name)
            elif all_file_names and global_idx < len(all_file_names):
                retrieved_files.append(all_file_names[global_idx])
            else:
                retrieved_files.append("")

        return retrieved_chunks, retrieved_indices, retrieved_files
    except Exception as e:
        st.error(f"Error retrieving chunks for query '{query}': {str(e)}")
        return [], [], []

def generate_answer(
    query: str,
    relevant_chunks: List[str],
    cohere_client,
    model: str = 'command-r-08-2024',
    source_names: Optional[List[str]] = None
) -> str:
    """
    Generate answer using Cohere API and retrieved chunks.
    """
    context = "\n".join(relevant_chunks)
    unique_sources = sorted(set(source_names)) if source_names else []
    source_prompt = (
        "Relevant policy documents: " + ", ".join(unique_sources) + ". "
        if unique_sources else ""
    )
    prompt = (
        "Use the provided context to answer the question. "
        "Reference the relevant policy documents by name when you cite their information. "
        f"{source_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    try:
        response = cohere_client.chat(model=model, message=prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return "Unable to generate answer."


def generate_policy_brief(
    query: str,
    relevant_chunks: List[str],
    cohere_client,
    model: str = 'command-r-08-2024',
    template: Optional[str] = None,
    template_instructions: Optional[str] = None,
    source_names: Optional[List[str]] = None,
    primary_policy: Optional[str] = None
) -> str:
    """Generate a structured policy brief using the retrieved context and optional template."""

    default_template = (
        "Policy Brief Title:\n"
        "Executive Summary:\n"
        "Background and Problem Definition:\n"
        "Key Findings:\n"
        "Policy Options and Analysis:\n"
        "Recommended Actions:\n"
        "Implementation Considerations:\n"
        "References or Supporting Evidence:\n"
    )

    template_text = template or default_template
    instruction_text = template_instructions.strip() if template_instructions else ""
    context = "\n".join(relevant_chunks)
    unique_sources = sorted(set(source_names)) if source_names else []
    source_prompt = (
        "You may reference the following policy documents, but only list their names under 'References or Supporting Evidence': "
        + ", ".join(unique_sources)
        + "."
        if unique_sources
        else ""
    )
    policy_focus = (
        f"Focus the brief on the policy titled '{primary_policy}'. "
        if primary_policy else ""
    )

    instructions = (
        "You are an expert policy analyst. Use the provided context to craft a concise, actionable policy brief. "
        "Populate every section of the template, synthesizing evidence-driven insights. "
        "Highlight implications, cite supporting facts from the context, and ensure recommendations are practical. "
        "Do NOT include inline citations or parentheses with policy names in any section. "
        "List policy titles only in the 'References or Supporting Evidence' section. "
        f"{policy_focus}"
        f"{source_prompt}"
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Template:\n{template_text}\n\n"
    )
    if instruction_text:
        prompt += f"Template Instructions:\n{instruction_text}\n\n"
    prompt += (
        f"Task:\n{instructions}\n\n"
        f"Policy Brief Request:\n{query}\n\n"
        "Complete Policy Brief:"
    )
    try:
        response = cohere_client.chat(model=model, message=prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating policy brief: {str(e)}")
        return "Unable to generate policy brief."
