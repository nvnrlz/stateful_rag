"""LLM providers for natural-language phrasing.

Design note (safety): clinical *control flow* — what to ask next, when to escalate
a red flag, when to stop — is handled deterministically by the triage engine, NOT
by a free-running LLM. The LLM, when configured, is used only to phrase questions
and summaries from structured, grounded inputs. This keeps the medically-relevant
decisions auditable and reproducible, and lets the whole system run offline with a
template responder when no LLM is configured.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...


class RuleBasedLLM(LLMProvider):
    """Sentinel 'no LLM' provider. The triage engine uses templates instead."""

    name = "rule_based"

    def generate(self, system: str, user: str) -> str:  # pragma: no cover
        raise RuntimeError("RuleBasedLLM does not generate text; use the template responder.")


class GeminiLLM(LLMProvider):
    """Google AI Studio (Gemini) chat via REST. Needs GOOGLE_API_KEY.

    Gemini *Flash* models are the right tier for low-latency receptionist phrasing.
    Note: model id must be a real Gemini Flash id (e.g. ``gemini-2.5-flash`` /
    ``gemini-2.0-flash``); there is no "3.5 flash".
    """

    _BASE = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None):
        self.name = model
        self._key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not self._key:
            raise RuntimeError("GOOGLE_API_KEY / GEMINI_API_KEY not set for GeminiLLM")
        import httpx
        try:
            import certifi
            self._client = httpx.Client(timeout=60, verify=certifi.where())
        except Exception:
            self._client = httpx.Client(timeout=60)

    def generate(self, system: str, user: str) -> str:
        from ._http import post_with_retry
        url = f"{self._BASE}/models/{self.name}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1024},
        }
        r = post_with_retry(self._client, url, params={"key": self._key}, json=body)
        data = r.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts).strip()


class OpenAILLM(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai not installed. pip install openai") from exc
        self._client = OpenAI()
        self.name = model

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - needs key
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class AnthropicLLM(LLMProvider):
    def __init__(self, model: str = "claude-3-5-sonnet-latest"):
        try:
            from anthropic import Anthropic
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("anthropic not installed. pip install anthropic") from exc
        self._client = Anthropic()
        self.name = model

    def generate(self, system: str, user: str) -> str:  # pragma: no cover - needs key
        msg = self._client.messages.create(
            model=self.name,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=0.2,
        )
        return "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")


def build_llm(provider: str, *, model: str = "") -> Optional[LLMProvider]:
    """Return an LLM provider, or ``None`` for rule-based (template) phrasing."""
    provider = (provider or "rule_based").lower()
    if provider == "rule_based":
        return None
    if provider in ("gemini", "google"):
        return GeminiLLM(model=model or "gemini-2.5-flash")
    if provider == "openai":
        return OpenAILLM(model=model or "gpt-4o-mini")
    if provider == "anthropic":
        return AnthropicLLM(model=model or "claude-3-5-sonnet-latest")
    raise ValueError(f"Unknown LLM provider: {provider!r}")
