"""
Semantic search over a notebook's FAISS index. Embeds the question with
the same provider used for ingestion, searches for the closest chunks,
and translates FAISS's raw positions back into real chunk metadata.
"""

import numpy as np
from backend.ingest.index_store import get_index


def vector_search(notebook_id: str, question: str, provider, top_k: int = 20) -> list[dict]:
    """
    Returns up to top_k chunks, each with its metadata plus a 'score'
    (lower = more similar, since FAISS's IndexFlatL2 measures distance).
    Returns an empty list if the notebook has no indexed documents yet.
    """
    stored = get_index(notebook_id)
    if stored is None:
        return []

    index, id_map = stored["index"], stored["id_map"]

    question_embedding = provider.embed([question])[0]
    query_vector = np.array([question_embedding], dtype="float32")

    k = min(top_k, index.ntotal)  # can't ask for more results than exist
    distances, positions = index.search(query_vector, k)

    results = []
    for dist, pos in zip(distances[0], positions[0]):
        if pos == -1:  # FAISS pads with -1 if fewer than k results exist
            continue
        chunk = id_map[pos].copy()
        chunk["score"] = float(dist)
        results.append(chunk)

    return results