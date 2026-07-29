"""
Turns chunks into embeddings and stores them in a FAISS index for search.
Checks Redis first so re-embedding identical text is never wasted work.
"""

import hashlib
import numpy as np
import faiss
from backend.ingest.redis_store import get_cached_embedding, cache_embedding

FAISS_DIMENSION = 384  # matches all-MiniLM-L6-v2's output size — must stay
                        # in sync with whichever embedding model is used


def _text_hash(text: str) -> str:
    """Short, stable key for caching — identical text always hashes the same."""
    return hashlib.md5(text.encode()).hexdigest()


def embed_chunks(chunks: list[dict], provider) -> list[list[float]]:
    """
    Returns one embedding per chunk, in the same order as `chunks`.
    Cached chunks skip the provider call entirely.
    """
    embeddings = [None] * len(chunks)
    to_compute = []       # (index, text) pairs not found in cache
    to_compute_texts = []

    for i, chunk in enumerate(chunks):
        h = _text_hash(chunk["text"])
        cached = get_cached_embedding(h)
        if cached is not None:
            embeddings[i] = cached
        else:
            to_compute.append(i)
            to_compute_texts.append(chunk["text"])

    if to_compute_texts:
        new_embeddings = provider.embed(to_compute_texts)
        for idx, text, emb in zip(to_compute, to_compute_texts, new_embeddings):
            embeddings[idx] = emb
            cache_embedding(_text_hash(text), emb)

    return embeddings


def build_faiss_index(chunks: list[dict], embeddings: list[list[float]]):
    """
    Builds a fresh FAISS index from chunks + their embeddings.
    Returns (index, id_map) — id_map maps FAISS's internal integer
    positions back to our chunk metadata, since FAISS itself only
    knows about vectors, not what they represent.
    """
    vectors = np.array(embeddings, dtype="float32")
    index = faiss.IndexFlatL2(FAISS_DIMENSION)
    index.add(vectors)

    id_map = {i: chunks[i] for i in range(len(chunks))}
    return index, id_map