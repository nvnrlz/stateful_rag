"""Pydantic request/response models for the chat API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    # The signed-disclosure requirement is enforced here: a session cannot start
    # without the tester acknowledging the test-only / privacy disclosure.
    consent_acknowledged: bool = Field(..., description="Tester accepted the test-only disclosure")
    principal: Optional[str] = Field(None, description="Opaque tester/session principal")


class StartResponse(BaseModel):
    session_id: str
    reply: str
    disclaimer: str


class ChatRequest(BaseModel):
    session_id: str
    text: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    action: str
    area: str
    turn: int
    pending_slot: Optional[str] = None
    citations: List[str] = []
    retrieval_route: Optional[str] = None
    red_flags: List[str] = []
    urgent_flags: List[str] = []
    intake: dict = {}
    concluded: bool
    disclaimer: str


class FeedbackRequest(BaseModel):
    session_id: str
    turn: int
    verdict: str = Field(..., description="correct | partially_correct | wrong | unsafe")
    reviewer_role: str = Field("mbbs", description="mbbs | ent_pg")
    reviewer_id: Optional[str] = None
    assistant_text: str = ""
    citations: List[str] = []
    comment: str = ""


class SessionFeedbackRequest(BaseModel):
    session_id: str
    overall_comment: str = ""
    enhancements: str = ""
    rating: Optional[int] = Field(None, ge=1, le=5)
    reviewer_role: str = Field("mbbs", description="mbbs | ent_pg")
    reviewer_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    knowledge_chunks: int
    embedding_model: str
    embedding_dim: int
    llm_provider: str
    offline: bool
