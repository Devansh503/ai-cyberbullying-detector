from __future__ import annotations
from typing import Dict, Any
from better_profanity import profanity
from .preprocessing import clean_text

profanity.load_censor_words()

REPLACEMENTS = {
    "hate": "dislike",
    "stupid": "not very thoughtful",
    "idiot": "person",
    "kill": "harm",
    "dumb": "unwise"
}


def rephrase_to_safe(text: str) -> Dict[str, Any]:
    raw = text
    cleaned = clean_text(text)
    censored = profanity.censor(raw)
    tokens = cleaned.split()
    replaced = [REPLACEMENTS.get(t, t) for t in tokens]
    suggestion = " ".join(replaced)
    if suggestion.strip() == cleaned.strip():
        suggestion = censored
    return {
        "original": raw,
        "censored": censored,
        "suggestion": suggestion
    }
