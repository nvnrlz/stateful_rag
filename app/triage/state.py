"""Per-session conversation state and a pluggable session store."""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Message:
    role: str          # "user" | "assistant"
    text: str
    ts: float = field(default_factory=time.time)


@dataclass
class SessionState:
    session_id: str
    principal: Optional[str] = None
    turn: int = 0
    complaint_area: Optional[str] = None     # ear | nose | throat | neck | unclear
    intake: Dict[str, object] = field(default_factory=dict)  # structured intake slots
    asked: List[str] = field(default_factory=list)        # gating slots already asked
    pending_slot: Optional[str] = None       # slot the last question targeted
    messages: List[Message] = field(default_factory=list)
    red_flagged: bool = False
    concluded: bool = False
    created_at: float = field(default_factory=time.time)

    def add_message(self, role: str, text: str) -> None:
        self.messages.append(Message(role=role, text=text))

    def transcript(self, last_n: Optional[int] = None) -> str:
        msgs = self.messages if last_n is None else self.messages[-last_n:]
        return "\n".join(f"{m.role}: {m.text}" for m in msgs)

    def user_text_so_far(self) -> str:
        return " ".join(m.text for m in self.messages if m.role == "user")


class SessionStore(ABC):
    @abstractmethod
    def get(self, session_id: str) -> Optional[SessionState]: ...

    @abstractmethod
    def save(self, state: SessionState) -> None: ...

    @abstractmethod
    def create(self, principal: Optional[str] = None) -> SessionState: ...

    @abstractmethod
    def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.RLock()

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, state: SessionState) -> None:
        with self._lock:
            self._sessions[state.session_id] = state

    def create(self, principal: Optional[str] = None) -> SessionState:
        state = SessionState(session_id=uuid.uuid4().hex, principal=principal)
        self.save(state)
        return state

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
