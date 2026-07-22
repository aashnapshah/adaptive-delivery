"""Small parsing helpers shared by the generation and judging code."""

from __future__ import annotations

import json
import re


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (tolerates code fences / surrounding prose)."""
    if not text or not text.strip():
        raise ValueError("empty model output")
    cleaned = re.sub(r"```(?:json)?|```", "", text)   # strip code fences
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in: {text[:160]}")
    return json.loads(m.group(0))
