"""
In-memory store mapping notebook_id -> (FAISS index, id_map).
NOT persisted to disk yet — restarting the backend clears this.
That's a known, deliberate limitation for this stage; disk/DB
persistence is a planned follow-up once the core pipeline works.
"""

_indexes = {}  # notebook_id -> {"index": faiss.Index, "id_map": {...}}


def save_index(notebook_id: str, index, id_map: dict):
    _indexes[notebook_id] = {"index": index, "id_map": id_map}


def get_index(notebook_id: str):
    return _indexes.get(notebook_id)