"""
Cross-encoder reranking: re-scores the top-20 hybrid results by actually
reading the question and each chunk together, which is more accurate
than comparing pre-computed embeddings — but too slow to run over every
chunk in the whole document, hence only applying it to the top-20.
"""

from sentence_transformers import CrossEncoder

_reranker = None  # module-level singleton, same pattern as the embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


def rerank(question: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Takes hybrid search's top-20 chunks, re-scores each against the
    question directly, and returns the top_k best. Returns immediately
    if given fewer chunks than top_k — nothing to narrow down.
    """
    if len(chunks) <= top_k:
        return chunks

    pairs = [[question, chunk["text"]] for chunk in chunks]
    scores = _get_reranker().predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    results = []
    for chunk, score in ranked[:top_k]:
        result = chunk.copy()
        result["rerank_score"] = float(score)
        results.append(result)

    return results