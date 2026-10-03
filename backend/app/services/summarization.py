"""Summarization with Gemini primary and extractive fallback."""

from __future__ import annotations

import logging
import re
from collections import Counter

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.services.gemini_client import generate_text, gemini_configured, get_gemini_model_name

logger = logging.getLogger(__name__)

# Unicode ranges used as a sanity check (not a full language detector)
_SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "te": (0x0C00, 0x0C7F),  # Telugu
}


def _looks_like_language(text: str, language: str) -> bool:
    """Heuristic: for Indic targets, require some chars in the expected script."""
    if language == "en":
        # Mostly Latin letters
        latin = sum(1 for ch in text if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
        return latin >= max(8, len(text) // 10)
    bounds = _SCRIPT_RANGES.get(language)
    if not bounds:
        return True
    lo, hi = bounds
    hits = sum(1 for ch in text if lo <= ord(ch) <= hi)
    return hits >= 5


def gemini_status() -> dict:
    """Backward-compatible helper used by health endpoint (enriched elsewhere)."""
    settings = get_settings()
    return {
        "configured": settings.gemini_configured,
        "model": get_gemini_model_name(),
    }


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?।؟])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 20]


def _extractive_summary(text: str, max_sentences: int = 6) -> str:
    """Rank sentences by word frequency (language-agnostic extractive summary)."""
    sentences = _split_sentences(text)
    if not sentences:
        cleaned = text.strip()
        if len(cleaned) <= 600:
            return cleaned
        return cleaned[:600].rsplit(" ", 1)[0] + "..."

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"\w+", text.lower())
    stop = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "is",
        "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "with", "as", "by", "from",
        "not", "no", "yes", "so", "if", "then", "than", "too", "very", "just", "about",
        "hai", "hain", "ka", "ki", "ke", "mein", "se", "aur", "ek", "yeh", "woh",
    }
    freq = Counter(w for w in words if w not in stop and len(w) > 2)
    if not freq:
        return " ".join(sentences[:max_sentences])

    scored: list[tuple[float, int, str]] = []
    for idx, sent in enumerate(sentences):
        tokens = re.findall(r"\w+", sent.lower())
        if not tokens:
            continue
        score = sum(freq.get(t, 0) for t in tokens) / len(tokens)
        score += max(0, 1.0 - idx * 0.02)
        scored.append((score, idx, sent))

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences]
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join(s for _, _, s in top_sorted)


def _gemini_summarize(transcript: str, output_language: str) -> tuple[str | None, dict | None]:
    """
    Ask Gemini to produce the final summary directly in the selected language.
    Returns (summary_or_none, structured_error_or_none).
    """
    if not gemini_configured():
        return None, {
            "status": None,
            "error_type": "NotConfigured",
            "message": "GEMINI_API_KEY is missing.",
        }

    lang_name = SUPPORTED_LANGUAGES.get(output_language, "English")
    prompt = f"""You are summarizing a podcast transcript for an Indian multilingual audience.

Write a concise but meaningful summary ENTIRELY in {lang_name}.
The entire output must be in {lang_name} script/language (not English, unless English was requested).

The summary MUST include:
1) Main topic
2) Important points discussed
3) Conclusions / key takeaways

Rules:
- Do NOT copy the transcript verbatim.
- Preserve names and important technical terms.
- Keep regional/cultural context.
- Length: about 120-220 words (or equivalent).
- Output only the summary text, no headings like "Summary:".

Transcript:
{transcript[:12000]}
"""
    result = generate_text(prompt, purpose="summarization")
    if result.ok and result.text:
        if not _looks_like_language(result.text, output_language):
            err = {
                "status": None,
                "error_type": "WrongLanguage",
                "message": (
                    f"Gemini returned text that does not appear to be in "
                    f"{SUPPORTED_LANGUAGES.get(output_language, output_language)}."
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
        "message": "Gemini summarization failed.",
    }


def summarize_transcript(
    transcript: str,
    output_language: str = "en",
    source_language: str = "en",
) -> tuple[str, str, str | None, str, dict | None]:
    """
    Return (summary_text, method, warning_message, summary_language, gemini_error).

    method: gemini | extractive
    summary_language: ISO code of the language the summary text is actually in
    """
    logger.info("Summarization started (target=%s, source=%s)", output_language, source_language)
    if not transcript or not transcript.strip():
        raise RuntimeError("Cannot summarize an empty transcript.")

    gemini_summary, gemini_err = _gemini_summarize(transcript, output_language)
    if gemini_summary:
        logger.info("Summary completed via Gemini in language=%s", output_language)
        return gemini_summary, "gemini", None, output_language, None

    # Keep fallback, but surface root cause clearly
    if gemini_err:
        logger.error(
            "[Gemini ERROR] summarization fallback engaged | status=%s type=%s message=%s",
            gemini_err.get("status"),
            gemini_err.get("error_type"),
            gemini_err.get("message"),
        )

    err_type = (gemini_err or {}).get("error_type", "")
    err_msg = (gemini_err or {}).get("message", "")
    status = (gemini_err or {}).get("status")

    if err_type == "NotConfigured" or "missing" in err_msg.lower():
        warning = (
            "Summary was generated with a local extractive method because the Gemini API key is missing. "
            "Set GEMINI_API_KEY in backend/.env for multilingual summaries."
        )
    elif status == 429 or "quota" in err_msg.lower() or "ResourceExhausted" in err_type:
        warning = (
            f"Gemini API quota was exceeded (status={status}). "
            "A local extractive summary was used instead. "
            "The extractive summary remains in the transcript language until translation succeeds."
        )
    else:
        warning = (
            f"Gemini summarization failed ({err_type or 'error'}"
            f"{f', status={status}' if status else ''}). "
            "A local extractive summary was used instead. "
            "See backend logs for [Gemini ERROR] details."
        )

    summary = _extractive_summary(transcript)
    # Extractive summary is in the source/transcript language
    logger.info("Summary completed via extractive fallback (language=%s)", source_language)
    return summary, "extractive", warning, source_language, gemini_err
