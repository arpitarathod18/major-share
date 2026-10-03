"""Shared Gemini client with structured logging and error reporting."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class GeminiError:
    status: int | None
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass
class GeminiResult:
    text: str | None
    model: str | None
    error: GeminiError | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text and self.text.strip())


def _safe_preview(text: str, limit: int = 80) -> str:
    """ASCII-safe preview for Windows consoles."""
    cleaned = re.sub(r"[^\x20-\x7E]+", "?", (text or "")[:limit])
    return cleaned


def _parse_gemini_exception(exc: Exception) -> GeminiError:
    status = getattr(exc, "code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    # google.api_core often puts status on .grpc_status / args
    msg = str(exc)
    if status is None:
        m = re.search(r"\b([45]\d{2})\b", msg)
        if m:
            status = int(m.group(1))
    error_type = type(exc).__name__
    # Strip any accidental key-like tokens
    sanitized = re.sub(r"(AIza[0-9A-Za-z_\-]{10,}|AQ\.[0-9A-Za-z_\-]{10,})", "[REDACTED]", msg)
    sanitized = sanitized[:800]
    return GeminiError(status=status, error_type=error_type, message=sanitized)


def gemini_configured() -> bool:
    return get_settings().gemini_configured


def get_gemini_model_name() -> str:
    settings = get_settings()
    return (settings.gemini_model or "gemini-2.5-flash").strip()


def _configure() -> tuple[Any, str] | tuple[None, None]:
    settings = get_settings()
    configured = settings.gemini_configured
    logger.info("[Gemini] API key configured: %s", str(configured).lower())
    if not configured:
        return None, None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model_name = get_gemini_model_name()
        logger.info("[Gemini] Model: %s", model_name)
        return genai, model_name
    except Exception as exc:  # noqa: BLE001
        err = _parse_gemini_exception(exc)
        logger.error(
            "[Gemini ERROR]\nStatus: %s\nType: %s\nMessage: %s",
            err.status,
            err.error_type,
            err.message,
        )
        return None, None


def generate_text(prompt: str, *, purpose: str = "generate") -> GeminiResult:
    """Call Gemini generateContent with structured logging."""
    genai, model_name = _configure()
    if genai is None or not model_name:
        return GeminiResult(
            text=None,
            model=None,
            error=GeminiError(
                status=None,
                error_type="NotConfigured",
                message="GEMINI_API_KEY is missing or Gemini SDK failed to initialize.",
            ),
        )

    logger.info("[Gemini] Request started (%s)", purpose)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = (getattr(response, "text", None) or "").strip()
        if not text:
            # Sometimes blocked / empty candidates
            err = GeminiError(
                status=None,
                error_type="EmptyResponse",
                message="Gemini returned an empty response (possibly blocked or unfinished).",
            )
            logger.error(
                "[Gemini ERROR]\nStatus: %s\nType: %s\nMessage: %s",
                err.status,
                err.error_type,
                err.message,
            )
            return GeminiResult(text=None, model=model_name, error=err)

        logger.info(
            "[Gemini] Request completed (%s) chars=%d preview=%s",
            purpose,
            len(text),
            _safe_preview(text),
        )
        return GeminiResult(text=text, model=model_name, error=None)
    except Exception as exc:  # noqa: BLE001
        err = _parse_gemini_exception(exc)
        logger.error(
            "[Gemini ERROR]\nStatus: %s\nType: %s\nMessage: %s",
            err.status,
            err.error_type,
            err.message,
        )
        return GeminiResult(text=None, model=model_name, error=err)


def probe_gemini(target_language: str = "kn") -> dict[str, Any]:
    """
    Tiny live check: ask Gemini to rewrite one sentence into the target language.
    Does not hardcode expected text; only checks non-empty response.
    """
    from app.config import SUPPORTED_LANGUAGES

    lang_name = SUPPORTED_LANGUAGES.get(target_language, "Kannada")
    prompt = (
        f"Summarize this sentence in {lang_name}. "
        "Output only the summary text, nothing else.\n\n"
        "Local language content improves access to education."
    )
    result = generate_text(prompt, purpose="probe")
    available = result.ok
    payload: dict[str, Any] = {
        "configured": gemini_configured(),
        "available": available,
        "model": result.model or get_gemini_model_name(),
        "probe_language": target_language,
    }
    if result.error:
        payload["error"] = result.error.to_dict()
    if available:
        payload["probe_chars"] = len(result.text or "")
    return payload
