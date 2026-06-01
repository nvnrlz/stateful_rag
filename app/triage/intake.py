"""Structured patient-intake schema and the 'minimum intake before DB' policy.

The receptionist must collect the basic characterisation of a complaint *before*
looking anything up — but it must also NOT re-ask things the patient already
said. This module defines the slots, which ones are required before a knowledge
lookup, and the (deterministic) decision of what is still missing.

The required set follows a standard symptom-characterisation frame (SOCRATES /
OPQRST) adapted for ENT triage:

    site (+ side) · main symptom · onset/duration · severity (+ impact) ·
    ≥1 associated symptom · progression · patient age band
"""

from __future__ import annotations

from typing import Dict, List, Optional

AREAS = ("ear", "nose", "throat", "neck")

# Canonical, allowed values per categorical slot (used to validate extraction).
ALLOWED = {
    "area": set(AREAS),
    "laterality": {"left", "right", "both"},
    "onset": {"sudden", "gradual"},
    "severity": {"mild", "moderate", "severe"},
    "progression": {"better", "worse", "same"},
    "age_band": {"adult", "child"},
    "prior_episodes": {"first", "recurrent"},
}

# All slot keys the extractor may populate.
SLOT_KEYS = [
    "area", "laterality", "symptom", "onset", "duration", "severity",
    "associated", "progression", "age_band", "triggers", "prior_episodes",
    "treatments_tried",
]


def new_intake() -> Dict[str, object]:
    intake: Dict[str, object] = {k: None for k in SLOT_KEYS}
    intake["associated"] = []
    return intake


def merge_intake(intake: Dict[str, object], updates: Dict[str, object]) -> None:
    """Merge extractor updates in place: fill blanks, never overwrite with null."""
    for key, val in (updates or {}).items():
        if key not in SLOT_KEYS or val in (None, "", "null"):
            continue
        if key == "associated":
            existing = list(intake.get("associated") or [])
            for item in (val if isinstance(val, list) else [val]):
                if item and item not in existing:
                    existing.append(item)
            intake["associated"] = existing
            continue
        if key in ALLOWED and str(val).lower() not in ALLOWED[key]:
            continue  # ignore out-of-vocabulary categorical values
        if key in ALLOWED:
            val = str(val).lower()
        if not intake.get(key):  # don't overwrite an already-filled slot
            intake[key] = val


def _filled(intake: Dict[str, object], slot: str) -> bool:
    if slot == "area":
        return intake.get("area") in AREAS
    if slot == "associated":
        return bool(intake.get("associated"))
    return bool(intake.get(slot))


# Question phrasing for each gating slot. The LLM responder rephrases these
# naturally; they remain the deterministic source of intent.
QUESTIONS = {
    "area": "Which part is affected — your ear, your nose, your throat, or your neck?",
    "symptom": ("What is the main problem you're noticing — for example pain, "
                "discharge, blockage, a change in hearing, bleeding, dizziness, or a "
                "change in your voice?"),
    "duration": "How long have you had this, and did it start suddenly or gradually?",
    "laterality": "Is it affecting your left side, your right side, or both?",
    "severity": ("How bad is it right now — mild, moderate, or severe — and is it "
                 "affecting your sleep or daily activities?"),
    "progression": "Since it started, is it getting better, getting worse, or staying about the same?",
    "age_band": "Just to note for our records — is the patient an adult or a child?",
}

ASSOC_QUESTIONS = {
    "ear": "Along with this, do you have any hearing change, ringing, ear discharge, or dizziness?",
    "nose": "Do you also have nasal blockage, discharge, bleeding, or a reduced sense of smell?",
    "throat": "Do you have pain on swallowing, a change in your voice, or a persistent cough?",
    "neck": "Is the lump painful or growing, and do you have any fever, weight loss, or night sweats?",
    "unclear": "Are there any other symptoms going along with this?",
}

# Order in which missing slots are requested (one per turn).
_ASK_ORDER = ["area", "symptom", "duration", "laterality", "severity",
              "associated", "progression", "age_band"]


def required_slots(area: Optional[str]) -> List[str]:
    """The slots that must be filled (or already asked) before a DB lookup."""
    req = ["symptom", "duration", "severity", "associated", "progression", "age_band"]
    if area not in AREAS:
        req = ["area"] + req
    elif area in ("ear", "nose", "neck"):
        req.append("laterality")
    return req


def missing_required(intake: Dict[str, object], area: Optional[str],
                     asked: List[str]) -> List[str]:
    """Ordered list of required slots that are neither filled nor already asked."""
    req = set(required_slots(area))
    return [s for s in _ASK_ORDER
            if s in req and not _filled(intake, s) and s not in asked]


def question_for(slot: str, area: Optional[str]) -> str:
    if slot == "associated":
        return ASSOC_QUESTIONS.get(area or "unclear", ASSOC_QUESTIONS["unclear"])
    return QUESTIONS.get(slot, "Could you tell me a little more about your symptom?")


def retrieval_query(intake: Dict[str, object], area: Optional[str]) -> str:
    """Build a rich knowledge-base query from the collected intake."""
    parts = [area or ""]
    for k in ("symptom", "laterality", "severity", "duration"):
        if intake.get(k):
            parts.append(str(intake[k]))
    assoc = intake.get("associated") or []
    parts.extend(str(a) for a in assoc)
    return " ".join(p for p in parts if p).strip()


def recap_lines(intake: Dict[str, object]) -> List[str]:
    """Human-readable recap of what was collected (for the triage summary)."""
    labels = {
        "symptom": "Main problem", "area": "Area", "laterality": "Side",
        "duration": "Duration", "onset": "Onset", "severity": "Severity",
        "progression": "Progression", "age_band": "Patient", "triggers": "Possible trigger",
        "prior_episodes": "Prior episodes", "treatments_tried": "Treatment tried",
    }
    lines = []
    for key, label in labels.items():
        val = intake.get(key)
        if val:
            lines.append(f"  • {label}: {val}")
    assoc = intake.get("associated") or []
    if assoc:
        lines.append("  • Associated: " + ", ".join(str(a) for a in assoc))
    return lines
