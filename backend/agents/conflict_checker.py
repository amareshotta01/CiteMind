"""
Flags disagreement across retrieved chunks before an answer is
synthesized. Runs as a lightweight LLM pass over the retrieved chunks,
asking specifically whether they conflict — separate from the main
synthesizer so a conflict check failure never blocks a normal answer.
"""

from backend.providers.router import generate_with_fallback

CONFLICT_SYSTEM_PROMPT = """You check whether a set of source excerpts contain any factual disagreement with each other (e.g. different numbers, dates, or claims about the same thing).

Respond in exactly this format:
CONFLICT: yes or no
DETAILS: if yes, briefly state what disagrees and which sources (by number) disagree. If no, write "none"."""


def check_conflicts(chunks: list[dict]) -> dict:
    if len(chunks) < 2:
        return {"has_conflict": False, "details": "none"}

    excerpts_text = "\n\n".join(
        f"[{i+1}] (page {c['page']}): {c['text']}"
        for i, c in enumerate(chunks)
    )

    response = generate_with_fallback(excerpts_text, system=CONFLICT_SYSTEM_PROMPT)
    response_upper = response.upper()

    has_conflict = "CONFLICT: YES" in response_upper

    details_line = ""
    for line in response.split("\n"):
        if line.strip().upper().startswith("DETAILS:"):
            details_line = line.split(":", 1)[1].strip()
            break

    return {"has_conflict": has_conflict, "details": details_line or "none"}


