"""
Groq implementation of LLMProvider.
Groq only serves generation (fast inference on open models) — no embedding
endpoint — so embed() reuses the same local sentence-transformers model
as GeminiProvider, via the shared helper.
"""

import os
from groq import Groq
from backend.providers.base import LLMProvider
from backend.providers.gemini_provider import _get_embedder


class GroqProvider(LLMProvider):

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set in .env")
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def embed(self, texts: list[str]) -> list[list[float]]:
        return _get_embedder().encode(texts).tolist()