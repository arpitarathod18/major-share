"""Translation service with Gemini (real LLM translation only)."""

from __future__ import annotations

import logging

from app.config import SUPPORTED_LANGUAGES
from app.services.gemini_client import generate_text, gemini_configured

logger = logging.getLogger(__name__)

_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "kn": (0x0C80, 0x0CFF),
    "te": (0x0C00, 0x0C7F),
}


def _looks_like_language(text: str, language: str) -> bool:
    if language == "en":
        latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        return latin >= max(8, len(text) // 10)
    bounds = _SCRIPT_RANGES.get(language)
    if not bounds:
        return True
    lo, hi = bounds
    hits = sum(1 for ch in text if lo <= ord(ch) <= hi)
    return hits >= 5


def _gemini_translate(text: str, source_lang: str, target_lang: str) -> tuple[str | None, dict | None]:
    if not gemini_configured():
        return None, {
            "status": None,
            "error_type": "NotConfigured",
            "message": "GEMINI_API_KEY is missing.",
        }
    if not text.strip():
        return text, None

    src = SUPPORTED_LANGUAGES.get(source_lang, source_lang)
    tgt = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

    prompt = f"""Translate the following text from {src} to {tgt}.

Rules:
- Output MUST be entirely in {tgt}.
- Preserve names and important technical terms.
- Preserve meaning and regional context.
- Do NOT translate speaker labels like "Speaker 1".
- Output only the translated text.

Text:
{text[:10000]}
"""
    result = generate_text(prompt, purpose="translation")
    if result.ok and result.text:
        if not _looks_like_language(result.text, target_lang):
            err = {
                "status": None,
                "error_type": "WrongLanguage",
                "message": (
                    f"Gemini translation does not appear to be in "
                    f"{SUPPORTED_LANGUAGES.get(target_lang, target_lang)}."
                ),
            }
            logger.error(
                "[Gemini ERROR]\nStatus: %s\nType: %s\nMessage: %s",
                err["status"],
                err["error_type"],
                err["message"],
            )
            return None, err
        return result.text, None
    return None, result.error.to_dict() if result.error else {
        "status": None,
        "error_type": "Unknown",
        "message": "Gemini translation failed.",
    }


def translate_text(
    text: str,
    source_language: str,
    target_language: str,
) -> tuple[str, str | None, bool, dict | None]:
    """
    Translate text if languages differ using Gemini.

    Returns:
      (result_text, warning_or_none, translated_ok, gemini_error_or_none)

    translated_ok is True only when the returned text is in the target language
    (either already matching, or successfully translated).
    """
    if not text:
        return text, None, True, None
    if source_language == target_language:
        return text, None, True, None

    translated, err = _gemini_translate(text, source_language, target_language)
    if translated:
        logger.info(
            "Translation completed: %s -> %s",
            source_language,
            target_language,
        )
        return translated, None, True, None

    if err:
        logger.error(
            "[Gemini ERROR] translation failed | status=%s type=%s message=%s",
            err.get("status"),
            err.get("error_type"),
            err.get("message"),
        )

    src_name = SUPPORTED_LANGUAGES.get(source_language, source_language)
    tgt_name = SUPPORTED_LANGUAGES.get(target_language, target_language)
    if not gemini_configured():
        warning = (
            f"Translation from {src_name} to {tgt_name} requires Gemini. "
            "GEMINI_API_KEY is missing. Summary is NOT claimed to be in the selected language."
        )
    else:
        status = (err or {}).get("status")
        err_type = (err or {}).get("error_type", "error")
        warning = (
            f"Translation to {tgt_name} failed ({err_type}"
            f"{f', status={status}' if status else ''}). "
            f"Summary remains in {src_name} and is NOT claimed to be {tgt_name}. "
            "See backend logs for [Gemini ERROR] details."
        )
    return text, warning, False, err
