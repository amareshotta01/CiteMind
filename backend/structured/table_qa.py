"""
Handles CSV/table data separately from the embedding pipeline. Tabular
data stays as an exact DataFrame — numeric questions get answered from
the real data, not a fuzzy semantic match against embedded row text.
"""

import pandas as pd
from backend.providers.router import generate_with_fallback

_tables = {}  # notebook_id -> {"filename": ..., "df": DataFrame}

TABLE_SYSTEM_PROMPT = """You answer questions about a data table using ONLY the table shown below. Give exact, precise answers based on the actual values — do not estimate or guess.

If the table doesn't contain enough information to answer, say so clearly."""


def store_table(notebook_id: str, filename: str, df: pd.DataFrame):
    """Called from /upload when a CSV is received."""
    _tables[notebook_id] = {"filename": filename, "df": df}


def has_table(notebook_id: str) -> bool:
    return notebook_id in _tables


def query_table(notebook_id: str, question: str) -> dict:
    """
    Returns {"answer": str, "citations": [...]}, same shape as
    synthesize_answer(), so /query can treat both paths interchangeably.
    """
    stored = _tables.get(notebook_id)
    if stored is None:
        return {"answer": "No table has been uploaded to this notebook yet.", "citations": []}

    df, filename = stored["df"], stored["filename"]

    # Small tables: send the whole thing. Larger ones: send a preview —
    # keeps the prompt a reasonable size without failing outright.
    table_text = df.to_string(index=False) if len(df) <= 200 else df.head(200).to_string(index=False)

    prompt = f"Table from '{filename}':\n\n{table_text}\n\nQuestion: {question}"
    answer = generate_with_fallback(prompt, system=TABLE_SYSTEM_PROMPT)

    return {
        "answer": answer,
        "citations": [{"doc": filename, "page": "table", "snippet": f"{len(df)} rows, columns: {', '.join(df.columns)}"}],
    }