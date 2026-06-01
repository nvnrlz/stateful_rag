"""Slot extraction from the patient's free-text conversation.

Design decision (per product owner): extraction is **LLM-only, no heuristic
fallback**. A regex guesser would silently produce low-quality intake on messy
real-patient language, which is worse than admitting the service is unavailable.
So if no LLM is configured, or the LLM call fails, the engine reports an
"unavailable" state instead of guessing.

The extractor only pulls *facts the patient stated*; it never invents details.
The engine still owns all control flow (what to ask, when to stop) — the LLM is a
structured information extractor, not a decision maker.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Optional

from ..providers.llm import LLMProvider

_SYSTEM = (
    "You extract structured ENT (ear/nose/throat) intake facts from a patient "
    "conversation. Return ONLY a compact JSON object — no prose, no code fences. "
    "Use ONLY information the patient explicitly stated; never infer or invent. "
    "If a field was not stated, use null (or [] for associated)."
)

_SCHEMA_HINT = (
    '{\n'
    '  "area": "ear|nose|throat|neck or null",\n'
    '  "laterality": "left|right|both or null",\n'
    '  "symptom": "short phrase of the main problem or null",\n'
    '  "onset": "sudden|gradual or null",\n'
    '  "duration": "short phrase e.g. \\"3 days\\" or null",\n'
    '  "severity": "mild|moderate|severe or null",\n'
    '  "associated": ["list of associated symptoms the patient mentioned"],\n'
    '  "progression": "better|worse|same or null",\n'
    '  "age_band": "adult|child or null",\n'
    '  "triggers": "short phrase e.g. \\"after swimming\\" or null",\n'
    '  "prior_episodes": "first|recurrent or null",\n'
    '  "treatments_tried": "short phrase or null"\n'
    '}'
)


class ExtractionUnavailable(Exception):
    """Raised when extraction cannot be performed (no LLM, or the call failed)."""


class SlotExtractor:
    def extract(self, transcript: str, intake: Dict[str, object]) -> Dict[str, object]:
        raise NotImplementedError


class GeminiSlotExtractor(SlotExtractor):
    """Extracts intake slots via the configured LLM (Gemini)."""

    def __init__(self, llm: LLMProvider):
        self._llm = llm

    def extract(self, transcript: str, intake: Dict[str, object]) -> Dict[str, object]:
        user = (
            "Conversation so far:\n" + transcript +
            "\n\nReturn the JSON object with this exact shape:\n" + _SCHEMA_HINT
        )
        try:
            raw = self._llm.generate(_SYSTEM, user)
        except Exception as exc:  # network / provider failure -> unavailable
            raise ExtractionUnavailable(str(exc)) from exc
        return _parse_json(raw)


def _parse_json(raw: str) -> Dict[str, object]:
    if not raw:
        return {}
    text = raw.strip()
    # Strip code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def build_extractor(llm: Optional[LLMProvider]) -> Optional[SlotExtractor]:
    """Return a Gemini extractor, or None when no LLM is configured (offline)."""
    return GeminiSlotExtractor(llm) if llm is not None else None
