"""Text-to-speech for English, Hindi, Kannada, and Telugu."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# edge-tts voice map for Indian languages
EDGE_VOICES = {
    "en": "en-IN-NeerjaNeural",
    "hi": "hi-IN-SwaraNeural",
    "kn": "kn-IN-SapnaNeural",
    "te": "te-IN-ShrutiNeural",
}

GTTS_LANG = {
    "en": "en",
    "hi": "hi",
    "kn": "kn",
    "te": "te",
}

SUPPORTED_TTS_LANGUAGES = ("en", "hi", "kn", "te")


def tts_supported_languages() -> list[str]:
    return list(SUPPORTED_TTS_LANGUAGES)


def tts_engine_available() -> bool:
    try:
        import edge_tts  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        try:
            import gtts  # noqa: F401

            return True
        except Exception:  # noqa: BLE001
            return False


def tts_status() -> dict:
    available = tts_engine_available()
    return {
        "available": available,
        "supported_languages": tts_supported_languages() if available else [],
        "engines": ["edge-tts", "gTTS"],
    }


def language_supported_for_tts(language: str) -> bool:
    return language in SUPPORTED_TTS_LANGUAGES


async def _edge_tts_save(text: str, language: str, output_path: Path) -> bool:
    try:
        import edge_tts

        voice = EDGE_VOICES.get(language)
        if not voice:
            logger.warning("edge-tts has no voice mapping for language=%s", language)
            return False
        communicate = edge_tts.Communicate(text=text[:4500], voice=voice)
        await communicate.save(str(output_path))
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("edge-tts failed: %s", exc)
        return False


def _gtts_save(text: str, language: str, output_path: Path) -> bool:
    try:
        from gtts import gTTS

        lang = GTTS_LANG.get(language)
        if not lang:
            logger.warning("gTTS has no language mapping for language=%s", language)
            return False
        tts = gTTS(text=text[:4500], lang=lang)
        tts.save(str(output_path))
        return output_path.exists() and output_path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("gTTS failed: %s", exc)
        return False


def synthesize_speech(
    text: str,
    language: str,
    output_path: str | Path,
) -> tuple[bool, str | None, str | None]:
    """
    Generate speech audio for the FINAL summary text.
    Returns (success, method_or_none, error_message_or_none).
    """
    logger.info("TTS started (lang=%s)", language)
    if not text or not text.strip():
        return False, None, "Cannot synthesize empty summary text."

    if not language_supported_for_tts(language):
        return (
            False,
            None,
            f"Text-to-speech does not support language '{language}'. "
            f"Supported: {', '.join(SUPPORTED_TTS_LANGUAGES)}.",
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Prefer edge-tts (supports en/hi/kn/te)
    try:
        ok = asyncio.run(_edge_tts_save(text, language, out))
        if ok:
            logger.info("TTS completed via edge-tts (lang=%s)", language)
            return True, "edge-tts", None
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            try:
                ok = loop.run_until_complete(_edge_tts_save(text, language, out))
            finally:
                loop.close()
            if ok:
                logger.info("TTS completed via edge-tts (lang=%s)", language)
                return True, "edge-tts", None
        except Exception as exc:  # noqa: BLE001
            logger.warning("edge-tts event loop failed: %s", exc)

    if _gtts_save(text, language, out):
        logger.info("TTS completed via gTTS (lang=%s)", language)
        return True, "gtts", None

    msg = (
        "Audio summary could not be generated. "
        "Text-to-speech engines (edge-tts / gTTS) were unavailable or failed. "
        "The text summary is still available."
    )
    logger.warning(msg)
    return False, None, msg
