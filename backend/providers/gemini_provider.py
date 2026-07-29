"""
Gemini implementation of LLMProvider.
Uses Gemini Flash for generation and a local sentence-transformers model
for embeddings (Gemini's embedding API has separate, tighter free-tier
limits — running embeddings locally avoids burning that quota).
"""

import os
from google import genai
from sentence_transformers import SentenceTransformer
from backend.providers.base import LLMProvider

_embedder = None  # loaded once, reused (loading it per-call is slow)


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


class GeminiProvider(LLMProvider):

    def __init__(self, model: str = "gemini-2.0-flash"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in .env")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={"system_instruction": system} if system else None,
        )
        return response.text

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _get_embedder().encode(texts).tolist()