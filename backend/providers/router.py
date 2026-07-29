"""
Picks which provider to use and handles fallback if one fails.
Nothing else in the app should import GeminiProvider/GroqProvider directly —
they go through get_provider() instead.
"""

from backend.providers.gemini_provider import GeminiProvider
from backend.providers.groq_provider import GroqProvider

_PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
}

_instances = {}  # cache constructed providers, avoid re-init on every call


def _get_instance(name: str):
    if name not in _instances:
        _instances[name] = _PROVIDERS[name]()
    return _instances[name]


def get_provider(preferred: str = "gemini"):
    """Return a working provider, preferring `preferred`, falling back to
    the other one if it fails to initialize (e.g. missing API key)."""
    try:
        return _get_instance(preferred)
    except Exception as e:
        fallback = "groq" if preferred == "gemini" else "gemini"
        print(f"[router] {preferred} unavailable ({e}), falling back to {fallback}")
        return _get_instance(fallback)


def generate_with_fallback(prompt: str, system: str = "", preferred: str = "gemini") -> str:
    """Call generate(), retrying with the other provider if the first one
    fails at request time (e.g. rate limit hit)."""
    order = [preferred, "groq" if preferred == "gemini" else "gemini"]
    last_error = None
    for name in order:
        try:
            provider = _get_instance(name)
            return provider.generate(prompt, system)
        except Exception as e:
            last_error = e
            print(f"[router] {name} failed ({e}), trying next provider")
    raise RuntimeError(f"All providers failed. Last error: {last_error}")