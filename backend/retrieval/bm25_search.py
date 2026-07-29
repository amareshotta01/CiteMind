"""
Keyword search over a notebook's chunks using BM25 — catches exact terms
(names, numbers, specific phrases) that dense vector search can miss,
since embeddings capture meaning but can blur over precise wording.
"""

from rank_bm25 import BM25Okapi
from backend.ingest.index_store import get_index


def bm25_search(notebook_id: str, question: str, top_k: int = 20) -> list[dict]:
    """
    Returns up to top_k chunks ranked by keyword relevance, each with
    a 'score' (higher = more relevant, opposite direction from FAISS's
    distance score — hybrid.py will need to account for this).
    Returns an empty list if the notebook has no indexed documents yet.
    """
    stored = get_index(notebook_id)
    if stored is None:
        return []

    id_map = stored["id_map"]
    chunks = list(id_map.values())

    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    tokenized_question = question.lower().split()
    scores = bm25.get_scores(tokenized_question)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in ranked[:top_k]:
        result = chunk.copy()
        result["score"] = float(score)
        results.append(result)

    return results