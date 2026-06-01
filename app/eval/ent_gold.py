"""ENT gold evaluation set.

These are original, author-written patient-style queries (no text from the
reference books). Relevance is judged automatically by a *concept-keyword proxy*:
a retrieved chunk is counted relevant if its text contains any of the short,
generic clinical anchor terms for that query. This is a pragmatic, reproducible
stand-in for manual chunk labelling and is good enough to (a) calibrate the drift
thresholds and (b) track retrieval-quality regressions. For the formal study,
replace/augment with clinician-labelled relevance judgements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class GoldQuery:
    query: str
    area: str                 # ear | nose | throat | neck
    anchors: List[str]        # generic concept terms; ANY match => relevant
    note: str = ""


GOLD_QUERIES: List[GoldQuery] = [
    # ---- EAR ----
    GoldQuery("I have ear pain and discharge for a few days", "ear",
              ["otitis", "discharge", "otorrhoea", "otorrhea", "tympanic", "otalgia"]),
    GoldQuery("my child keeps pulling the ear and has a fever", "ear",
              ["otitis media", "otitis", "tympanic", "middle ear"]),
    GoldQuery("I suddenly lost hearing in one ear", "ear",
              ["sensorineural", "sudden", "hearing loss", "deafness"]),
    GoldQuery("there is constant ringing in my ears", "ear",
              ["tinnitus"]),
    GoldQuery("the room spins and I lose my balance", "ear",
              ["vertigo", "vestibular", "labyrinth", "meniere", "menier"]),
    GoldQuery("itchy ear with blocked feeling and wax", "ear",
              ["cerumen", "wax", "otitis externa", "external ear"]),
    GoldQuery("gradual hearing loss over years in both ears", "ear",
              ["sensorineural", "presbycusis", "conductive", "hearing loss"]),
    GoldQuery("ear fullness and popping when I fly", "ear",
              ["eustachian", "barotrauma", "middle ear", "pressure"]),

    # ---- NOSE ----
    GoldQuery("blocked and runny nose for weeks with sneezing", "nose",
              ["rhinitis", "allergic", "nasal obstruction", "congestion", "sneezing"]),
    GoldQuery("frequent heavy nosebleeds", "nose",
              ["epistaxis", "bleeding", "little's area", "kiesselbach"]),
    GoldQuery("facial pain and thick yellow nasal discharge", "nose",
              ["sinusitis", "rhinosinusitis", "sinus", "paranasal"]),
    GoldQuery("I cannot smell anything anymore", "nose",
              ["anosmia", "olfact", "smell"]),
    GoldQuery("one-sided nasal blockage with crusting", "nose",
              ["septum", "deviated", "polyp", "nasal obstruction"]),
    GoldQuery("nasal blockage and snoring at night", "nose",
              ["turbinate", "nasal obstruction", "adenoid", "snoring"]),
    GoldQuery("recurrent sinus infections not improving with tablets", "nose",
              ["chronic", "sinusitis", "rhinosinusitis", "sinus"]),
    GoldQuery("watery nasal discharge after a head injury", "nose",
              ["csf", "rhinorrhoea", "rhinorrhea", "cerebrospinal"]),

    # ---- THROAT ----
    GoldQuery("sore throat with pain on swallowing and fever", "throat",
              ["tonsillitis", "pharyngitis", "tonsil", "pharynx"]),
    GoldQuery("my voice has been hoarse for several weeks", "throat",
              ["hoarseness", "laryngitis", "larynx", "vocal cord", "vocal fold"]),
    GoldQuery("feeling of a lump in the throat", "throat",
              ["globus", "pharyngeal", "dysphagia"]),
    GoldQuery("difficulty swallowing solid foods", "throat",
              ["dysphagia", "swallowing", "oesophag", "esophag"]),
    GoldQuery("repeated tonsil infections every year", "throat",
              ["tonsillitis", "tonsillectomy", "tonsil"]),
    GoldQuery("noisy breathing and a harsh sound in my child", "throat",
              ["stridor", "laryngomalacia", "airway", "larynx"]),
    GoldQuery("white spots on the tonsils with bad breath", "throat",
              ["tonsil", "exudate", "tonsillolith", "tonsillitis"]),

    # ---- NECK ----
    GoldQuery("a lump in my neck that has been growing", "neck",
              ["neck mass", "lymph node", "lymphadenopathy", "swelling", "neck"]),
    GoldQuery("swollen glands in the neck with a sore throat", "neck",
              ["lymphadenopathy", "lymph node", "cervical", "reactive"]),
    GoldQuery("painless neck swelling with weight loss and night sweats", "neck",
              ["lymphoma", "malignan", "metasta", "neck mass"]),
    GoldQuery("midline neck swelling that moves when I swallow", "neck",
              ["thyroglossal", "thyroid", "midline", "neck"]),
    GoldQuery("swelling below the jaw worse during meals", "neck",
              ["salivary", "submandibular", "sialadenitis", "calculus", "stone"]),
]


# Multi-turn pairs for drift calibration: (first complaint, follow-up).
# Same-area pairs SHOULD remain a cache hit; cross-area pairs SHOULD break the cache.
GOLD_PAIRS_SAME: List[Tuple[str, str]] = [
    ("I have ear pain and discharge", "is my hearing affected and why does it ache"),
    ("blocked runny nose with sneezing", "will an antihistamine help my nasal allergy"),
    ("sore throat and painful swallowing", "are my tonsils infected and is it serious"),
    ("a growing lump in my neck", "should I be worried about this neck swelling"),
    ("ringing and fullness in my ear", "what causes this tinnitus and ear blockage"),
    ("facial pain with thick nasal discharge", "is this a sinus infection in my paranasal sinuses"),
    ("hoarse voice for weeks", "could this be a problem with my vocal cords or larynx"),
    ("dizziness and room spinning", "is this vertigo coming from my inner ear balance"),
]

GOLD_PAIRS_CROSS: List[Tuple[str, str]] = [
    ("I have ear pain and discharge", "I also have a growing lump in my neck"),
    ("blocked runny nose with sneezing", "now my voice is hoarse and throat hurts"),
    ("sore throat and painful swallowing", "by the way my ear is ringing constantly"),
    ("a growing lump in my neck", "and my nose is completely blocked with sneezing"),
    ("ringing and fullness in my ear", "also I have a sore throat and trouble swallowing"),
    ("facial pain with thick nasal discharge", "and there is a swelling in my neck"),
    ("hoarse voice for weeks", "my ear has been painful with discharge too"),
    ("dizziness and room spinning", "separately my tonsils have white spots"),
]

# (turn1, turn2, expected_label) where label True => should stay cached (on-topic).
GOLD_PAIRS: List[Tuple[str, str, bool]] = (
    [(a, b, True) for a, b in GOLD_PAIRS_SAME]
    + [(a, b, False) for a, b in GOLD_PAIRS_CROSS]
)
