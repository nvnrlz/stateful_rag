"""Deterministic ENT question plan and complaint-area detection.

Keeping the *what to ask next* logic here (not in an LLM) makes the clinical flow
auditable and reproducible — a requirement for a tool reviewed by doctors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

AREAS = ("ear", "nose", "throat", "neck")

_AREA_TERMS: Dict[str, Set[str]] = {
    "ear": {"ear", "hearing", "deaf", "tinnitus", "ringing", "vertigo", "dizzy",
            "dizziness", "earache", "otitis", "balance", "otalgia"},
    "nose": {"nose", "nasal", "sinus", "smell", "congestion", "runny", "sneez",
             "rhinitis", "polyp", "blocked nose", "stuffy", "epistaxis", "nosebleed"},
    "throat": {"throat", "tonsil", "voice", "hoarse", "swallow", "pharyng", "cough",
               "sore throat", "larynx", "snoring", "snore", "phlegm"},
    "neck": {"neck", "lump", "mass", "lymph", "node", "swelling in neck", "swollen gland"},
}

# Routing / department suggestion per area (receptionist deliverable).
DEPARTMENT = {
    "ear": "ENT (Otology) outpatient clinic",
    "nose": "ENT (Rhinology) outpatient clinic",
    "throat": "ENT (Laryngology / Head & Neck) outpatient clinic",
    "neck": "ENT Head & Neck outpatient clinic",
    "unclear": "General ENT outpatient clinic",
}


@dataclass
class Question:
    slot: str
    text: str
    areas: Optional[Set[str]] = None   # None => applies to any area


QUESTION_PLAN: List[Question] = [
    Question("onset", "How long have you had this, and did it come on suddenly or gradually?"),
    Question("laterality", "Is it affecting one side or both sides?", {"ear", "nose", "neck"}),
    Question("severity", "How bad is it right now — mild, moderate, or severe — and is it "
                          "affecting your sleep or daily activities?"),
    Question("assoc_ear", "Along with this, do you have any hearing change, ringing, ear "
                          "discharge, or dizziness?", {"ear"}),
    Question("assoc_nose", "Do you also have nasal blockage, discharge, bleeding, or a reduced "
                           "sense of smell?", {"nose"}),
    Question("assoc_throat", "Do you have pain on swallowing, a change in your voice, or a "
                             "persistent cough?", {"throat"}),
    Question("assoc_neck", "Is the lump painful or growing, and have you had any weight loss, "
                           "night sweats, or fever?", {"neck"}),
    Question("fever", "Do you have a fever or feel generally unwell?"),
    Question("triggers", "Did anything seem to trigger it — a recent cold, water exposure, "
                         "injury, or allergies?"),
]


def detect_area(text: str, current: Optional[str] = None) -> str:
    low = (text or "").lower()
    scores = {area: sum(1 for term in terms if term in low) for area, terms in _AREA_TERMS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return current or "unclear"
    return best


def next_question(area: str, asked: List[str]) -> Optional[Question]:
    for q in QUESTION_PLAN:
        if q.slot in asked:
            continue
        if q.areas is None or area in q.areas:
            return q
    return None


def applicable_slots(area: str) -> List[str]:
    return [q.slot for q in QUESTION_PLAN if q.areas is None or area in q.areas]
