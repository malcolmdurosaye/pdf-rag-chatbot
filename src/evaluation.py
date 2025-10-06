import json
import streamlit as st
import sqlite3
from typing import List, Tuple, Any
from .rag_pipeline import retrieve_chunks

def init_eval_db(db_path: str = 'data/rag.db') -> sqlite3.Connection:
    """
    Initialize SQLite table for evaluation results.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            precision REAL,
            mrr REAL,
            retrieved_indices TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    return conn

def generate_ground_truth(chunks: List[str], cohere_client, ground_truth_path: str) -> List[dict]:
    """
    Generate ground truth (query-chunk pairs) using Cohere API, with strict rate limit handling for trial keys.
    """
    import time
    ground_truth = []
    batch_size = 10  # Cohere trial key: 10 calls/minute
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start+batch_size]
        for i, chunk in enumerate(batch):
            prompt = f"Summarize this text in one sentence, then generate a question it answers:\n\nText: {chunk}\n\nOutput format: {{ \"summary\": \"...\", \"query\": \"...\" }}"
            response = None
            for attempt in range(3):
                try:
                    response = cohere_client.chat(model='command-r-08-2024', message=prompt)
                    result = json.loads(response.text)
                    if "query" in result:
                        ground_truth.append({"query": result["query"], "relevantIndices": [batch_start + i]})
                    else:
                        st.warning(f"Response for chunk {batch_start + i} missing 'query' key. Raw response: {response.text}")
                    break
                except Exception as e:
                    if hasattr(response, 'status_code') and response.status_code == 429:
                        st.warning(f"Rate limit hit for chunk {batch_start + i}. Waiting 60 seconds before retrying...")
                        time.sleep(60)
                        continue
                    raw_text = response.text if response and hasattr(response, 'text') else None
                    st.warning(f"Failed to parse ground truth for chunk {batch_start + i}: {str(e)}. Raw response: {raw_text}")
                    break
            time.sleep(7)  # Add delay between calls to avoid hitting rate limit
        if batch_start + batch_size < len(chunks):
            st.info("Waiting 60 seconds to respect Cohere trial key rate limit...")
            time.sleep(60)
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    return ground_truth

def save_ground_truth(ground_truth: List[dict], ground_truth_path: str) -> None:
    """
    Save ground truth to JSON.
    """
    with open(ground_truth_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

def evaluate_retrieval(
    ground_truth: List[dict], 
    index, 
    chunks: List[str], 
    embeddings, 
    cohere_client
) -> Tuple[float, float, List[Any]]:
    """
    Evaluate retrieval with Precision@3 and MRR, save to SQLite.
    """
    conn = init_eval_db()
    cursor = conn.cursor()
    total_precision = 0
    total_mrr = 0
    k = 3
    results = []
    for entry in ground_truth:
        query = entry["query"]
        relevant_indices = entry["relevantIndices"]
        _, retrieved_indices, _ = retrieve_chunks(query, index, chunks, embeddings, cohere_client, k)
        relevant_retrieved = len(set(retrieved_indices) & set(relevant_indices))
        precision = relevant_retrieved / k
        total_precision += precision
        mrr = 0
        for rank, idx in enumerate(retrieved_indices, 1):
            if idx in relevant_indices:
                mrr = 1 / rank
                break
        total_mrr += mrr
        results.append({"query": query, "precision": precision, "mrr": mrr, "retrieved_indices": retrieved_indices})
        # Save to SQLite
        cursor.execute(
            'INSERT INTO evaluation_results (query, precision, mrr, retrieved_indices, timestamp) VALUES (?, ?, ?, ?, datetime("now"))',
            (query, precision, mrr, json.dumps(retrieved_indices))
        )
    conn.commit()
    conn.close()
    avg_precision = total_precision / len(ground_truth) if ground_truth else 0
    avg_mrr = total_mrr / len(ground_truth) if ground_truth else 0
    return avg_precision, avg_mrr, results
