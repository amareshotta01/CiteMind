"""
Merges BM25 (keyword) and vector (semantic) search results into one
ranked list. The two searches score things on different, incompatible
scales (BM25: higher=better, FAISS distance: lower=better), so both
get normalized to a common 0-1 scale before combining.
"""

from backend.retrieval.vector_search import vector_search
from backend.retrieval.bm25_search import bm25_search


def _normalize(scores: list[float], higher_is_better: bool) -> list[float]:
    """Min-max normalize scores to 0-1, flipping direction if needed so
    1.0 always means 'best match' regardless of the original scale."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:  # all identical — avoid divide-by-zero, treat as neutral
        return [0.5] * len(scores)
    normalized = [(s - lo) / (hi - lo) for s in scores]
    return normalized if higher_is_better else [1 - n for n in normalized]


def hybrid_search(notebook_id: str, question: str, provider, top_k: int = 20,
                   vector_weight: float = 0.5) -> list[dict]:
    """
    Combines vector + BM25 results, deduplicating by chunk_id and
    blending normalized scores using vector_weight (0-1). A chunk found
    by only one method still gets ranked using just that method's score.
    """
    vector_results = vector_search(notebook_id, question, provider, top_k=top_k)
    bm25_results = bm25_search(notebook_id, question, top_k=top_k)

    vector_scores = _normalize([r["score"] for r in vector_results], higher_is_better=False)
    bm25_scores = _normalize([r["score"] for r in bm25_results], higher_is_better=True)

    combined = {}  # chunk_id -> {"chunk": ..., "vector_score": ..., "bm25_score": ...}

    for r, score in zip(vector_results, vector_scores):
        combined[r["chunk_id"]] = {"chunk": r, "vector_score": score, "bm25_score": 0.0}

    for r, score in zip(bm25_results, bm25_scores):
        if r["chunk_id"] in combined:
            combined[r["chunk_id"]]["bm25_score"] = score
        else:
            combined[r["chunk_id"]] = {"chunk": r, "vector_score": 0.0, "bm25_score": score}

    ranked = []
    for entry in combined.values():
        final_score = (vector_weight * entry["vector_score"] +
                        (1 - vector_weight) * entry["bm25_score"])
        chunk = entry["chunk"].copy()
        chunk["score"] = final_score
        ranked.append(chunk)

    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked[:top_k]