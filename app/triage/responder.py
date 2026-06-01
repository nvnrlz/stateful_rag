"""Turns structured triage decisions + grounding into patient-facing text.

Two responders share one interface:

- ``TemplateResponder`` (offline default): deterministic, fully auditable text.
- ``LLMResponder``: uses a configured LLM purely to *phrase* the same structured
  inputs more naturally. It is constrained to ENT triage and to the supplied
  grounding.

Both keep any quoted reference text very short and always attached to a citation
(source + page). The conversation deliverable is triage/routing and recaps — not
reproductions of the reference books.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..providers.llm import LLMProvider
from ..safety.guardrails import URGENT_ADVICE
from .intake import recap_lines
from .plan import DEPARTMENT
from .state import SessionState

# Hard cap on any single quoted snippet to avoid reproducing source text.
_SNIPPET_CHARS = 180


def _short_snippet(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:_SNIPPET_CHARS] + ("…" if len(text) > _SNIPPET_CHARS else "")


def format_citations(grounding: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    cites = []
    seen = set()
    for g in grounding[:limit]:
        c = g.get("citation") or g.get("source", "reference")
        if c not in seen:
            seen.add(c)
            cites.append(c)
    return cites


class TemplateResponder:
    name = "template"

    def greet(self) -> str:
        return ("Hello, I'm the ENT triage assistant for this test. "
                "Please tell me what ear, nose, or throat problem you're experiencing.")

    def ask(self, state: SessionState, question_text: str, grounding) -> str:
        return question_text

    def refuse(self, message: str) -> str:
        return message

    def escalate(self, advice: str) -> str:
        return advice

    def summarize(self, state: SessionState, grounding: List[Dict[str, Any]], urgent_flags=None) -> str:
        area = state.complaint_area or "unclear"
        dept = DEPARTMENT.get(area, DEPARTMENT["unclear"])
        lines: List[str] = []
        lines.append("Thank you. Here's a summary of what you've told me:")
        lines.extend(recap_lines(state.intake or {}))
        if urgent_flags:
            lines.append("\n" + URGENT_ADVICE)
            lines.append(f"This sounds like it relates to the **{area}** and should be "
                         f"seen **urgently** at the **{dept}**.")
        else:
            lines.append(f"\nThis sounds like it relates to the **{area}**. "
                         f"For a test like this, the appropriate next step would be the **{dept}**.")
        cites = format_citations(grounding)
        if cites:
            lines.append("\nReference material consulted: " + "; ".join(cites) + ".")
        lines.append("\nWould you like to add anything, or is there another ENT concern?")
        return "\n".join(lines)


class LLMResponder:
    name = "llm"

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    _SYSTEM = (
        "You are an ENT (ear/nose/throat) triage receptionist for a hospital, used "
        "ONLY in a controlled test. You never diagnose and never prescribe. You ask "
        "one short, clear question at a time to understand the patient's ENT problem, "
        "or you give a brief triage summary and which ENT clinic to visit. Keep "
        "replies concise and in plain language. Use ONLY the provided reference notes "
        "for any clinical statement, cite them as (source, page), and never quote more "
        "than a few words from them. If information is missing, ask for it rather than "
        "guessing."
    )

    def _ground_block(self, grounding: List[Dict[str, Any]]) -> str:
        if not grounding:
            return "(no reference notes retrieved)"
        parts = []
        for g in grounding[:3]:
            cite = g.get("citation", g.get("source", "reference"))
            parts.append(f"- ({cite}) {_short_snippet(g.get('content',''))}")
        return "\n".join(parts)

    def greet(self) -> str:
        return TemplateResponder().greet()

    def refuse(self, message: str) -> str:
        return message

    def escalate(self, advice: str) -> str:
        return advice

    def ask(self, state: SessionState, question_text: str, grounding) -> str:
        user = (
            f"Conversation so far:\n{state.transcript(last_n=8)}\n\n"
            f"Reference notes:\n{self._ground_block(grounding or [])}\n\n"
            f"Ask the patient this next, rephrased naturally and empathetically as ONE "
            f"short question: \"{question_text}\""
        )
        try:
            return self.llm.generate(self._SYSTEM, user).strip() or question_text
        except Exception:
            return question_text  # fail safe to the deterministic question

    def summarize(self, state: SessionState, grounding: List[Dict[str, Any]], urgent_flags=None) -> str:
        area = state.complaint_area or "unclear"
        dept = DEPARTMENT.get(area, DEPARTMENT["unclear"])
        urgency = ""
        if urgent_flags:
            urgency = (" IMPORTANT: these symptoms have persisted long enough to need an "
                       "URGENT (not emergency) ENT review — say so clearly and recommend an "
                       "urgent appointment.")
        user = (
            f"Conversation so far:\n{state.transcript()}\n\n"
            f"Reference notes:\n{self._ground_block(grounding)}\n\n"
            f"Give a brief triage summary: recap the key symptoms in one or two sentences, "
            f"state that this relates to the {area}, and recommend visiting the {dept}.{urgency} "
            f"Cite reference notes as (source, page) where you used them. Do NOT diagnose. "
            f"End by asking if they have anything to add."
        )
        try:
            out = self.llm.generate(self._SYSTEM, user).strip()
            return out or TemplateResponder().summarize(state, grounding, urgent_flags)
        except Exception:
            return TemplateResponder().summarize(state, grounding, urgent_flags)


def build_responder(llm: Optional[LLMProvider]):
    return TemplateResponder() if llm is None else LLMResponder(llm)
