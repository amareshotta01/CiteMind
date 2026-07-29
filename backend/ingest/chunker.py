"""
Splits loaded PDF page text into smaller chunks, tracking exactly where
each chunk came from (doc_id, page, char_offset). This metadata is what
makes citation enforcement possible later — without it, we could only
ever cite "somewhere in this document," which is the exact NotebookLM
problem this project exists to fix.
"""

import uuid

CHUNK_SIZE = 800       # characters per chunk — small enough for precise citations,
                        # large enough to keep sentences from being cut mid-thought
CHUNK_OVERLAP = 100     # characters shared between consecutive chunks, so an answer
                        # spanning a chunk boundary doesn't lose context


def chunk_pages(doc_id: str, pages: list[dict]) -> list[dict]:
    """
    pages: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    Returns a flat list of chunks:
    [{"chunk_id": "...", "doc_id": "...", "page": 1, "char_offset": 0, "text": "..."}]
    """
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()
            if chunk_text:  # skip empty slices (can happen at the very end)
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "doc_id": doc_id,
                    "page": page["page"],
                    "char_offset": start,
                    "text": chunk_text,
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP  # step forward, leaving overlap
    return chunks