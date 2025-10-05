import sqlite3
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional

def init_db(db_path: str = 'data/rag.db') -> sqlite3.Connection:
    """
    Initialize SQLite database with embeddings and evaluation tables.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            chunk TEXT,
            embedding BLOB
        )
    ''')
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

def save_chunks_and_embeddings(conn: sqlite3.Connection, file_name: str, chunks: List[str], embeddings: np.ndarray) -> None:
    """
    Save chunks and embeddings to SQLite with file_name metadata.
    """
    cursor = conn.cursor()
    cursor.execute('DELETE FROM embeddings WHERE file_name = ?', (file_name,))  # Clear existing for this file
    for chunk, emb in zip(chunks, embeddings):
        emb_blob = emb.tobytes()
        cursor.execute('INSERT INTO embeddings (file_name, chunk, embedding) VALUES (?, ?, ?)', (file_name, chunk, emb_blob))
    conn.commit()

def load_chunks_and_embeddings(conn: sqlite3.Connection) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Load all chunks and embeddings from SQLite (all files).
    """
    cursor = conn.cursor()
    cursor.execute('SELECT file_name, chunk, embedding FROM embeddings')
    rows = cursor.fetchall()
    chunks = [row[1] for row in rows]
    embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in rows]
    file_names = [row[0] for row in rows]  # Track file names
    return chunks, np.vstack(embeddings) if embeddings else np.array([]), file_names

def load_chunks_for_file(conn: sqlite3.Connection, file_name: str) -> Tuple[List[str], np.ndarray]:
    """
    Load chunks and embeddings for a specific file.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT chunk, embedding FROM embeddings WHERE file_name = ?', (file_name,))
    rows = cursor.fetchall()
    chunks = [row[0] for row in rows]
    embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
    return chunks, np.vstack(embeddings) if embeddings else np.array([])

def get_file_names(conn: sqlite3.Connection) -> List[str]:
    """
    Get all unique file names from embeddings.
    """
    cursor = conn.cursor()
    cursor.execute('SELECT DISTINCT file_name FROM embeddings ORDER BY file_name')
    return [row[0] for row in cursor.fetchall()]