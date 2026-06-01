"""The triage engine: safety → intake extraction → questioning → grounded routing.

Per-turn order (safety-first, deterministic control):

1. **Red-flag emergency check** — overrides everything, stops the flow.
2. **Scope check** (first turn) — steer non-ENT input back on topic.
3. **Slot extraction** — read everything the patient has said so far into a
   structured intake (LLM-only; if unavailable, say so — never guess).
4. **Minimum-intake gate** — if any *required* basic detail is still missing,
   ask exactly one missing question (never re-asking what was already given).
   Only once the minimum intake is complete do we hit the knowledge base.
5. **Grounded summary + routing**, with urgent (non-emergency) flags surfaced.

This both stops a layman's sparse "my ear hurts" from going to the DB prematurely
*and* lets a detailed opening message go straight to the DB in one turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..safety.guardrails import (
    EDUCATIONAL_DISCLAIMER,
    SCOPE_REFUSAL,
    detect_red_flags,
    detect_urgent_flags,
    is_in_scope,
)
from .extractor import ExtractionUnavailable, SlotExtractor
from .intake import (
    merge_intake,
    missing_required,
    new_intake,
    question_for,
    retrieval_query,
)
from .plan import detect_area
from .responder import TemplateResponder, format_citations
from .state import SessionState

OFFLINE_MESSAGE = (
    "I'm sorry — the intake service needed to understand your symptoms accurately "
    "is currently unavailable, so I can't continue this triage right now. Please "
    "try again shortly, or contact the clinic directly."
)
TRANSIENT_MESSAGE = (
    "I'm having trouble processing that right now. Could you please repeat or "
    "rephrase your last message?"
)


@dataclass
class TurnResult:
    assistant_text: str
    action: str                       # greet | ask | escalate | refuse | summary | unavailable
    area: str = "unclear"
    pending_slot: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    grounding: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_route: Optional[str] = None
    red_flags: List[str] = field(default_factory=list)
    urgent_flags: List[str] = field(default_factory=list)
    intake: Dict[str, Any] = field(default_factory=dict)
    concluded: bool = False
    turn: int = 0
    disclaimer: str = EDUCATIONAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.assistant_text,
            "action": self.action,
            "area": self.area,
            "pending_slot": self.pending_slot,
            "citations": self.citations,
            "retrieval_route": self.retrieval_route,
            "red_flags": self.red_flags,
            "urgent_flags": self.urgent_flags,
            "intake": self.intake,
            "concluded": self.concluded,
            "turn": self.turn,
            "disclaimer": self.disclaimer,
        }


class TriageEngine:
    def __init__(self, retriever, responder=None, *, extractor: Optional[SlotExtractor] = None,
                 max_turns: int = 8):
        self.retriever = retriever
        self.responder = responder or TemplateResponder()
        self.extractor = extractor
        self.max_turns = max_turns

    # -- grounding -----------------------------------------------------------
    def _retrieve(self, state: SessionState, query: str):
        if self.retriever is None:
            return [], None
        try:
            docs = self.retriever.retrieve(
                query=query, session_id=state.session_id,
                current_turn=state.turn, principal=state.principal,
            )
            route = docs[0].get("_route") if docs and isinstance(docs[0], dict) else None
            return docs, route
        except Exception:
            return [], None

    # -- main entry ----------------------------------------------------------
    def handle(self, state: SessionState, user_text: str) -> TurnResult:
        state.turn += 1
        state.add_message("user", user_text)
        if not state.intake:
            state.intake = new_intake()

        # 1) Red-flag emergency check — always first.
        flags = detect_red_flags(state.user_text_so_far())
        if flags:
            state.red_flagged = True
            state.concluded = True
            advice = flags[0].advice
            state.add_message("assistant", advice)
            return TurnResult(advice, "escalate", area=state.complaint_area or "unclear",
                              red_flags=[f.name for f in flags], concluded=True, turn=state.turn)

        # 2) Scope check on the first substantive message.
        if state.turn == 1 and not is_in_scope(user_text):
            state.add_message("assistant", SCOPE_REFUSAL)
            return TurnResult(SCOPE_REFUSAL, "refuse", turn=state.turn)

        # 3) Extract intake from everything said so far (LLM-only, no fallback).
        if self.extractor is None:
            state.add_message("assistant", OFFLINE_MESSAGE)
            return TurnResult(OFFLINE_MESSAGE, "unavailable", turn=state.turn)
        try:
            updates = self.extractor.extract(state.transcript(), state.intake)
        except ExtractionUnavailable:
            state.add_message("assistant", OFFLINE_MESSAGE)
            return TurnResult(OFFLINE_MESSAGE, "unavailable", turn=state.turn)
        except Exception:
            return TurnResult(TRANSIENT_MESSAGE, "unavailable", turn=state.turn)
        merge_intake(state.intake, updates)

        area = state.intake.get("area") or detect_area(user_text, state.complaint_area)
        state.complaint_area = area

        urgent = detect_urgent_flags(state.user_text_so_far())
        urgent_names = [f.name for f in urgent]
        intake_snapshot = dict(state.intake)

        # 4) Minimum-intake gate: ask only what's still missing (never re-ask).
        missing = missing_required(state.intake, area, state.asked)
        if missing and state.turn < self.max_turns:
            slot = missing[0]
            state.asked.append(slot)
            state.pending_slot = slot
            qtext = question_for(slot, area if area in ("ear", "nose", "throat", "neck") else "unclear")
            text = self.responder.ask(state, qtext, [])
            state.add_message("assistant", text)
            return TurnResult(text, "ask", area=area, pending_slot=slot, urgent_flags=urgent_names,
                              intake=intake_snapshot, turn=state.turn)

        # 5) Minimum intake complete -> NOW hit the knowledge base and summarise.
        query = retrieval_query(state.intake, area) or user_text
        grounding, route = self._retrieve(state, query)
        citations = format_citations(grounding)
        state.concluded = True
        text = self.responder.summarize(state, grounding, urgent_flags=urgent)
        state.add_message("assistant", text)
        return TurnResult(text, "summary", area=area, citations=citations, grounding=grounding,
                          retrieval_route=route, urgent_flags=urgent_names, intake=intake_snapshot,
                          concluded=True, turn=state.turn)

    def start(self, state: SessionState) -> TurnResult:
        if not state.intake:
            state.intake = new_intake()
        text = self.responder.greet()
        state.add_message("assistant", text)
        return TurnResult(text, "greet", turn=0)
