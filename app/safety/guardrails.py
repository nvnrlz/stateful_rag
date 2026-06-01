"""Clinical safety guardrails for the ENT receptionist.

Two jobs:

1. **Red-flag escalation.** Certain ENT presentations are emergencies. When any
   are detected in the conversation we short-circuit the normal questioning flow
   and tell the user to seek emergency care immediately, regardless of cache or
   retrieval state. This is deterministic and always runs first.

2. **Scope control + disclaimer.** The tool is an ENT *receptionist/triage* test
   aid, not a diagnostic device. Out-of-scope or non-medical inputs are politely
   refused, and every response carries an educational-use disclaimer.

These rules are intentionally conservative (high sensitivity): for a triage front
door, over-escalating is far safer than missing an emergency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

EDUCATIONAL_DISCLAIMER = (
    "⚠️ This is an experimental ENT triage assistant used for TESTING ONLY. "
    "It does not provide a diagnosis and must not be used for real medical "
    "decisions. For any real concern, consult a qualified doctor."
)

SCOPE_REFUSAL = (
    "I can only help with ear, nose, and throat (ENT) related symptoms in this "
    "test. Could you describe an ENT-related concern?"
)

EMERGENCY_ADVICE = (
    "🚑 Based on what you've described, this may be a medical EMERGENCY. "
    "Please seek immediate emergency care / call your local emergency number or "
    "go to the nearest emergency department now. Do not wait for this assistant."
)


@dataclass
class RedFlag:
    name: str
    advice: str
    matched: str


# Each rule: (flag name, list of regex patterns). Patterns are lowercase-matched.
_RED_FLAG_RULES = [
    # Unambiguous airway-emergency signs. General "breathing difficulty" is handled
    # separately below so it can exclude nasal-only blockage (see _airway_breathing).
    ("airway_compromise", [
        r"stridor", r"chok(e|ing)", r"gasp", r"turning blue", r"blue lips",
        r"cyanos", r"can'?t breathe at all", r"stopped breathing",
    ]),
    ("foreign_body_airway", [
        # Genuine ingestion/inhalation emergencies — NOT bare "something stuck in
        # throat", which is usually globus sensation (handled as urgent, not 911).
        r"swallowed.*(battery|button battery|coin|magnet|sharp)",
        r"inhaled (a|an|some|something|object|foreign)",
        r"(object|something) (stuck|lodged).{0,30}(can'?t breathe|choking|wind ?pipe|airway)",
        r"choking on", r"foreign body.{0,20}(airway|wind ?pipe)",
    ]),
    ("severe_epistaxis", [
        r"(heavy|severe|won'?t stop|can'?t stop|profuse).*(nose ?bleed|bleeding from .*nose)",
        r"nose ?bleed.*(won'?t|can'?t) stop", r"bleeding heavily.*nose",
    ]),
    ("post_op_bleed", [
        r"bleeding after (tonsil|adenoid|throat|nose|sinus) (surgery|operation|removal)",
        r"post.?tonsillectomy bleed",
    ]),
    ("deep_neck_infection", [
        r"can('?t| ?not) (open (my )?mouth|swallow)", r"drooling", r"swollen (neck|throat).*fever",
        r"ludwig", r"unable to swallow (saliva|spit)",
    ]),
    ("orbital_complication", [
        r"(eye|vision).*(swelling|swollen|bulging|double vision|loss).*(sinus|nose)",
        r"sinus.*(eye|vision) (swelling|problem|loss)", r"proptosis",
    ]),
    ("neuro_vertigo", [
        r"(vertigo|dizzy|dizziness).*(weak|numb|slurred|face droop|can'?t walk|double vision)",
        r"sudden.*(slurred speech|face droop|one.sided weakness)",
    ]),
]

_COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in _RED_FLAG_RULES]

# ENT scope keywords — broad on purpose; used to politely steer non-ENT chatter.
_ENT_TERMS = (
    "ear", "hearing", "deaf", "tinnitus", "vertigo", "dizzy", "dizziness", "nose",
    "nasal", "sinus", "smell", "snoring", "snore", "throat", "tonsil", "voice",
    "hoarse", "swallow", "neck", "lymph", "nasopharyn", "larynx", "cough", "cold",
    "congestion", "runny", "block", "pain", "bleed", "discharge", "lump", "mass",
    "fever", "vomit", "headache", "polyp", "allergy", "sneez", "earache", "otitis",
    "rhinitis", "pharyng", "stridor", "breath", "balance",
)


# Breathing difficulty, in many phrasings. Treated as an airway emergency UNLESS
# it is clearly nasal-only ("can't breathe through my nose" = blockage, not 911).
_BREATH_DIFFICULTY = re.compile(
    r"(can'?t|cannot|can not|trouble|hard to|difficult(?:y)? (?:to )?|struggl\w* to|"
    r"unable to|short of|laboured|labored|noisy)\s*breath\w*|short of breath|"
    r"breathless|can'?t catch my breath"
)
# Nasal-blockage context that should NOT be read as airway compromise.
_NASAL_BREATH = re.compile(
    r"breath\w*\s*(through|out of|via)\s*(my |the )?(nose|nostril)|"
    r"(nose|nostril)\s*(is|are|feels?)?\s*(block|congest|stuff)"
)
_THROAT_AIRWAY_CTX = re.compile(
    r"throat|swallow|stridor|chok|wind ?pipe|airway|voice|drool|chest|gasp"
)
# Epiglottitis / deep-neck-space infection cluster — an airway-threatening
# emergency that can present without the patient saying the word "stridor".
_EPIGLOTTITIS = re.compile(
    r"drool|can'?t swallow (?:my )?(?:saliva|spit)|cannot swallow|"
    r"muffled voice|hot[- ]potato voice|can'?t (?:open|swallow)|tripod"
)


def _airway_emergency(low: str) -> str | None:
    """Return a flag name if breathing difficulty / epiglottitis is present."""
    breath = _BREATH_DIFFICULTY.search(low)
    nasal_only = _NASAL_BREATH.search(low) and not _THROAT_AIRWAY_CTX.search(low)
    if breath and not nasal_only:
        return "airway_compromise"
    # Throat danger cluster: a swallowing/voice/drooling sign with fever or a
    # throat complaint suggests epiglottitis / deep-neck infection.
    if _EPIGLOTTITIS.search(low) and re.search(r"throat|swallow|voice|fever|neck", low):
        return "airway_throat_emergency"
    return None


def detect_red_flags(text: str) -> List[RedFlag]:
    low = (text or "").lower()
    flags: List[RedFlag] = []
    for name, patterns in _COMPILED:
        for pat in patterns:
            m = pat.search(low)
            if m:
                flags.append(RedFlag(name=name, advice=EMERGENCY_ADVICE, matched=m.group(0)))
                break
    extra = _airway_emergency(low)
    if extra and extra not in {f.name for f in flags}:
        flags.insert(0, RedFlag(name=extra, advice=EMERGENCY_ADVICE, matched="airway"))
    return flags


# --- Urgent (NOT emergency) referral flags ------------------------------------
# These do not warrant calling an emergency number, but should be seen by an ENT
# specialist soon (e.g. a persistently hoarse voice can indicate laryngeal
# pathology and needs laryngoscopy). They are surfaced in the triage summary and
# bump the recommended urgency — they do not stop the conversation.
URGENT_ADVICE = (
    "⏱️ These symptoms have lasted long enough that you should see an ENT "
    "specialist soon — an urgent (but not emergency) appointment — for proper "
    "examination."
)

_DURATION_WEEKS = r"(?:[3-9]|[1-9]\d)\s*week|weeks|month|months|fortnight"

_URGENT_RULES = [
    ("persistent_hoarseness", [
        rf"(hoarse|hoarseness|raspy|rough voice|voice (?:change|loss|getting weaker)|losing my voice).{{0,40}}(?:{_DURATION_WEEKS})",
        rf"(?:{_DURATION_WEEKS}).{{0,40}}(hoarse|raspy|voice (?:change|loss))",
    ]),
    ("persistent_neck_lump", [
        rf"(neck (?:lump|mass|swelling|node)|lump in (?:my )?neck).{{0,40}}(?:{_DURATION_WEEKS})",
    ]),
    ("persistent_unilateral_nasal", [
        rf"(one[- ]sided|unilateral).{{0,30}}(nose|nasal|nostril).{{0,30}}(block|obstruct|bleed)",
    ]),
    # Sudden sensorineural hearing loss (incl. "muffled" / overnight onset) is
    # time-critical: needs ENT review within ~24-48h (steroid window). Urgent,
    # not a 911 emergency.
    ("sudden_hearing_loss", [
        r"sudden(ly)?\s*(deaf|lost (my )?hearing|can'?t hear|muffled hearing)",
        r"(woke up|overnight|this morning).{0,40}(muffled|reduced|lost|can'?t hear|deaf)",
        r"(muffled|reduced) hearing.{0,30}(sudden|overnight|woke|morning)",
        r"hearing.{0,15}(gone|dropped).{0,20}(sudden|overnight)",
    ]),
]
_URGENT_COMPILED = [(n, [re.compile(p) for p in pats]) for n, pats in _URGENT_RULES]


def detect_urgent_flags(text: str) -> List[RedFlag]:
    low = (text or "").lower()
    flags: List[RedFlag] = []
    for name, patterns in _URGENT_COMPILED:
        for pat in patterns:
            m = pat.search(low)
            if m:
                flags.append(RedFlag(name=name, advice=URGENT_ADVICE, matched=m.group(0)))
                break
    return flags


def is_in_scope(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in _ENT_TERMS)
