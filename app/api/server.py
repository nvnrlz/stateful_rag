"""FastAPI chat server for the ENT receptionist beta.

Endpoints:
- GET  /health           — liveness + corpus/provider info
- POST /session/start    — consent-gated session creation, returns greeting
- POST /chat             — one conversational turn
- POST /feedback         — reviewer (student-doctor) verdict capture
- GET  /feedback/summary — running accuracy summary for the test

The IVR channel can reuse this same service layer later; only the transport
differs. For now only the chat transport is exposed, as requested.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from ..config import load_config
from ..feedback import Feedback, SessionFeedback
from ..service import ReceptionistService, SessionBusy
from ..triage.agent import EDUCATIONAL_DISCLAIMER
from .schemas import (
    ChatRequest,
    ChatResponse,
    FeedbackRequest,
    HealthResponse,
    SessionFeedbackRequest,
    StartRequest,
    StartResponse,
)

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def create_app(service: ReceptionistService | None = None) -> FastAPI:
    cfg = load_config()
    svc = service or ReceptionistService(cfg)
    app = FastAPI(title="ENT AI Receptionist (Test)", version="0.1.0")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Never let an unexpected error kill the worker or leave the UI hanging —
        # return a clean, friendly message instead.
        return JSONResponse(
            status_code=500,
            content={
                "reply": ("Sorry, something went wrong on our side. Please try sending "
                          "your message again."),
                "action": "error",
                "disclaimer": EDUCATIONAL_DISCLAIMER,
                "detail": "internal_error",
            },
        )

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(os.path.join(_STATIC_DIR, "index.html"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            knowledge_chunks=svc.knowledge_count(),
            embedding_model=svc.embedder.name,
            embedding_dim=svc.embedder.dim,
            llm_provider=svc.cfg.llm_provider,
            offline=svc.cfg.is_offline(),
        )

    @app.post("/session/start", response_model=StartResponse)
    def start(req: StartRequest) -> StartResponse:
        if not req.consent_acknowledged:
            raise HTTPException(
                status_code=403,
                detail="Consent to the test-only disclosure is required to begin.",
            )
        state, result = svc.start_session(principal=req.principal)
        return StartResponse(
            session_id=state.session_id, reply=result.assistant_text, disclaimer=result.disclaimer
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        try:
            result = svc.message(req.session_id, req.text)
        except KeyError:
            raise HTTPException(status_code=404, detail="Unknown session_id. Start a session first.")
        except SessionBusy:
            # Duplicate/impatient resend while the previous message is in flight.
            raise HTTPException(
                status_code=409,
                detail="Still processing your previous message — please wait a moment before sending again.",
            )
        return ChatResponse(
            session_id=req.session_id,
            reply=result.assistant_text,
            action=result.action,
            area=result.area,
            turn=result.turn,
            pending_slot=result.pending_slot,
            citations=result.citations,
            retrieval_route=result.retrieval_route,
            red_flags=result.red_flags,
            urgent_flags=result.urgent_flags,
            intake=result.intake,
            concluded=result.concluded,
            disclaimer=result.disclaimer,
        )

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        try:
            svc.record_feedback(Feedback(
                session_id=req.session_id, turn=req.turn, verdict=req.verdict,
                reviewer_role=req.reviewer_role, reviewer_id=req.reviewer_id,
                assistant_text=req.assistant_text, citations=req.citations, comment=req.comment,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"status": "recorded"}

    @app.post("/session/feedback")
    def session_feedback(req: SessionFeedbackRequest) -> dict:
        try:
            svc.record_session_feedback(SessionFeedback(
                session_id=req.session_id, overall_comment=req.overall_comment,
                enhancements=req.enhancements, rating=req.rating,
                reviewer_role=req.reviewer_role, reviewer_id=req.reviewer_id,
            ))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"status": "recorded"}

    @app.get("/feedback/summary")
    def feedback_summary() -> dict:
        return svc.feedback.summary()

    return app
