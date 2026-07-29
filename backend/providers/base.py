"""
Abstract interface every LLM provider must implement.

Why this exists: nothing else in the app should ever call Gemini or Groq
directly. Retrieval, agents, everything talks to this interface instead —
so swapping providers later is a one-line config change, not a rewrite.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt (with optional system instruction) and return the
        model's text response."""
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of text chunks into embedding vectors."""
        raise NotImplementedError