"""Capture reviewer (student-doctor) judgments for the evaluation.

The whole point of the beta is to learn whether answers are correct. This logs,
as append-only JSONL, each reviewer verdict (correct / partially correct / wrong)
against the session and turn, plus the conversation snapshot and retrieval
citations, so results can be analysed afterwards. No raw patient identity is
required — sessions are opaque ids.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

VALID_VERDICTS = {"correct", "partially_correct", "wrong", "unsafe"}


@dataclass
class Feedback:
    session_id: str
    turn: int
    verdict: str
    reviewer_role: str = "mbbs"          # mbbs | ent_pg
    reviewer_id: Optional[str] = None
    assistant_text: str = ""
    citations: List[str] = field(default_factory=list)
    comment: str = ""
    ts: float = field(default_factory=time.time)
    type: str = "turn_verdict"

    def validate(self) -> None:
        if self.verdict not in VALID_VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(VALID_VERDICTS)}")


@dataclass
class SessionFeedback:
    """Overall, end-of-session feedback the tester leaves about the whole test."""

    session_id: str
    overall_comment: str = ""
    enhancements: str = ""
    rating: Optional[int] = None         # optional 1-5 overall rating
    reviewer_role: str = "mbbs"
    reviewer_id: Optional[str] = None
    ts: float = field(default_factory=time.time)
    type: str = "session_feedback"

    def validate(self) -> None:
        if self.rating is not None and not (1 <= self.rating <= 5):
            raise ValueError("rating must be between 1 and 5")
        if not (self.overall_comment.strip() or self.enhancements.strip() or self.rating):
            raise ValueError("provide at least a comment, enhancement, or rating")


class FeedbackLog:
    """Append-only JSONL feedback log (swap for a DB table in production)."""

    def __init__(self, path: str = "data/feedback.jsonl"):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def record(self, fb: Feedback) -> None:
        fb.validate()
        self._append(asdict(fb))

    def record_session(self, sf: SessionFeedback) -> None:
        sf.validate()
        self._append(asdict(sf))

    def _append(self, obj: Dict[str, Any]) -> None:
        with self._lock:
            with open(self.path, "a") as f:
                f.write(json.dumps(obj) + "\n")

    def all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def summary(self) -> Dict[str, Any]:
        rows = self.all()
        verdicts = [r for r in rows if r.get("type", "turn_verdict") == "turn_verdict"]
        session_fb = [r for r in rows if r.get("type") == "session_feedback"]
        counts: Dict[str, int] = {}
        for r in verdicts:
            counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
        total = len(verdicts)
        return {
            "total": total,
            "by_verdict": counts,
            "accuracy": round(counts.get("correct", 0) / total, 4) if total else None,
            "session_feedback_count": len(session_fb),
        }
