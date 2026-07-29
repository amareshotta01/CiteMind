"""
Loaders: extract raw text/rows from uploaded files.
Job stops at "give me text + basic structure" — chunking and embedding
happen in separate files, so each step does exactly one thing.
"""

import io
import pandas as pd
from pypdf import PdfReader


def load_pdf(file_bytes: bytes) -> list[dict]:
    """Extract text from a PDF, one entry per page.
    Returns: [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}, ...]
    Page numbers are kept here because citations later need to point to
    an exact page, not just 'somewhere in this document'.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():  # skip blank pages, nothing useful to chunk
            pages.append({"page": i, "text": text})
    return pages


def load_csv(file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV into a DataFrame as-is.
    Unlike PDFs, CSVs are NOT turned into prose text here — they stay
    structured, because numeric questions (e.g. 'what was Q3 revenue?')
    need exact lookups, not fuzzy embedding search. This DataFrame goes
    to structured/table_qa.py later, not the embedding pipeline.
    """
    return pd.read_csv(io.BytesIO(file_bytes))


def load_document(filename: str, file_bytes: bytes):
    """Single entry point — picks the right loader based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return {"type": "pdf", "content": load_pdf(file_bytes)}
    elif ext == "csv":
        return {"type": "csv", "content": load_csv(file_bytes)}
    else:
        raise ValueError(f"Unsupported file type: .{ext} — only PDF and CSV are supported")