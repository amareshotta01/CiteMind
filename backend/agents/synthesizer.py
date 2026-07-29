"""
Turns retrieved chunks into a grounded answer with enforced citations.
The model is instructed to only answer from the given chunks and to
say so explicitly if it can't — this is the actual mechanism behind
'citation enforcement', not just a UI label.
"""

from backend.providers.router import generate_with_fallback

SYSTEM_PROMPT = """You are a document Q&A assistant. Answer the user's question using ONLY the provided source excerpts below. Each excerpt is labeled with a source number.

Rules:
- If the excerpts don't contain enough information to answer, say so clearly instead of guessing.
- After each claim you make, cite the source number it came from, like [1] or [2].
- Do not use any outside knowledge beyond what's in the excerpts."""


def synthesize_answer(question: str, chunks: list[dict]) -> dict:
    """
    Returns {"answer": str, "citations": [{"doc": ..., "page": ..., "snippet": ...}]}
    citations list mirrors the source numbers used in the answer text.
    """
    if not chunks:
        return {
            "answer": "I don't have any documents to search yet — please upload one first.",
            "citations": [],
        }

    excerpts_text = "\n\n".join(
        f"[{i+1}] (page {c['page']}): {c['text']}"
        for i, c in enumerate(chunks)
    )

    prompt = f"Source excerpts:\n\n{excerpts_text}\n\nQuestion: {question}"

    answer = generate_with_fallback(prompt, system=SYSTEM_PROMPT)

    citations = [
        {"doc": c.get("doc_id", "unknown"), "page": c["page"], "snippet": c["text"][:150]}
        for c in chunks
    ]

    return {"answer": answer, "citations": citations}