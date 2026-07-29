"""
Redis-backed cache for chunk content and embeddings.
Not the source of truth (MongoDB is) — this only exists to avoid
re-fetching from Mongo or re-computing embeddings on every query.
"""

import os
import json
import redis
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True,
        )
    return _client


def cache_chunk(chunk_id: str, text: str, metadata: dict, ttl_seconds: int = 86400):
    """Store chunk text + metadata for fast lookup by ID. Expires after
    ttl_seconds (default 1 day) so stale cache doesn't grow forever."""
    _get_client().set(f"chunk:{chunk_id}", json.dumps({"text": text, "metadata": metadata}), ex=ttl_seconds)


def get_chunk(chunk_id: str) -> dict | None:
    """Fast path: read a chunk by ID without touching MongoDB."""
    raw = _get_client().get(f"chunk:{chunk_id}")
    return json.loads(raw) if raw else None


def cache_embedding(text_hash: str, embedding: list[float], ttl_seconds: int = 604800):
    """Cache an embedding by a hash of its source text, so re-uploading
    the same content skips recomputation. Kept longer (7 days) since
    embeddings for identical text never go stale."""
    _get_client().set(f"embed:{text_hash}", json.dumps(embedding), ex=ttl_seconds)


def get_cached_embedding(text_hash: str) -> list[float] | None:
    raw = _get_client().get(f"embed:{text_hash}")
    return json.loads(raw) if raw else None