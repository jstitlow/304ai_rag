import ollama
import numpy as np


def get_embedding(text: str, model: str = "nomic-embed-text:latest"):
    """
    Generate normalized embedding vector using Ollama Python SDK.
    Compatible with pgvector (list[float]).
    """

    if not text or not text.strip():
        raise ValueError("Empty text passed to embedding")

    response = ollama.embed(
        model=model,
        input=text
    )

    # ---- HANDLE SDK OBJECT ----
    if not hasattr(response, "embeddings") or not response.embeddings:
        raise ValueError(f"Ollama returned no embeddings: {response}")

    vec = response.embeddings[0]   # <-- THE FIX

    # normalize for cosine similarity
    vec = np.array(vec, dtype=float)
    vec = vec / np.linalg.norm(vec)

    return vec.tolist()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """
    Split text into overlapping chunks.
    This dramatically improves retrieval quality in RAG.
    """

    if not text:
        return []

    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks