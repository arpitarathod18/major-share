"""Quality validation module for transcription, diarization, and language results."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

_INDIC_RANGES = {
    "hi": (0x0900, 0x097F),  # Devanagari
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "te": (0x0C00, 0x0C7F),  # Telugu
}


@dataclass
class ValidationReport:
    passed: bool
    warnings: list[str]
    errors: list[str]
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "warnings": self.warnings,
            "errors": self.errors,
            "details": self.details,
        }


def validate_results(
    audio_duration: float,
    detected_language: str,
    language_confidence: float,
    segments: list[Any],
    diarization_available: bool,
    speakers: list[str],
) -> ValidationReport:
    """Perform comprehensive quality validation on final processed podcast data."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "timestamp_validation": "PASS",
        "speaker_validation": "PASS",
        "language_validation": "PASS",
        "segment_count": len(segments),
        "speaker_count": len(speakers),
    }

    # 1. Transcript emptiness check
    if not segments:
        errors.append("Transcript is empty.")
        details["transcript_validation"] = "FAIL"
    else:
        details["transcript_validation"] = "PASS"

    # 2. Timestamp range and chronological order checks
    prev_end = 0.0
    for idx, seg in enumerate(segments):
        start = float(getattr(seg, "start", 0.0) if not isinstance(seg, dict) else seg.get("start", 0.0))
        end = float(getattr(seg, "end", 0.0) if not isinstance(seg, dict) else seg.get("end", 0.0))
        text = str(getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")).strip()

        if start < 0:
            errors.append(f"Segment {idx} has negative start time: {start:.3f}s")
            details["timestamp_validation"] = "FAIL"
        if end <= start:
            errors.append(f"Segment {idx} has invalid duration (start={start:.3f}s, end={end:.3f}s)")
            details["timestamp_validation"] = "FAIL"
        if end > audio_duration + 2.0:
            warnings.append(
                f"Segment {idx} end time ({end:.3f}s) slightly exceeds audio duration ({audio_duration:.3f}s)"
            )
        if idx > 0 and start < prev_end - 0.5:
            warnings.append(f"Segment {idx} start ({start:.3f}s) occurs before previous segment end ({prev_end:.3f}s)")
        prev_end = end

        # 3. Unicode script validation
        seg_lang = str(getattr(seg, "language", "") if not isinstance(seg, dict) else seg.get("language", ""))
        if seg_lang in _INDIC_RANGES and text:
            lo, hi = _INDIC_RANGES[seg_lang]
            indic_chars = sum(1 for ch in text if lo <= ord(ch) <= hi)
            if indic_chars == 0 and len(text) > 15 and re.search(r"[a-zA-Z]", text):
                warnings.append(f"Segment {idx} is labeled {SUPPORTED_LANGUAGES.get(seg_lang, seg_lang)} but appears to contain Latin script text.")

    # 4. Speaker validation
    if diarization_available:
        unassigned = sum(
            1 for seg in segments if not (getattr(seg, "speaker", None) if not isinstance(seg, dict) else seg.get("speaker"))
        )
        if unassigned > 0:
            warnings.append(f"{unassigned} segments could not be matched to an identified speaker turn.")
        if len(speakers) > 8:
            warnings.append(f"High number of distinct speakers detected ({len(speakers)}). Check for over-clustering.")
        if len(speakers) <= 1:
            details["speaker_validation"] = "PASS (Single speaker detected)"
        else:
            details["speaker_validation"] = f"PASS (Multi-speaker: {len(speakers)} detected)"
    else:
        details["speaker_validation"] = "SKIPPED (Diarization unavailable)"

    # 5. Language detection confidence check
    if detected_language not in SUPPORTED_LANGUAGES:
        errors.append(f"Detected language '{detected_language}' is not one of the supported languages (en, hi, kn, te).")
        details["language_validation"] = "FAIL"
    elif language_confidence < 0.35:
        warnings.append(f"Language detection confidence is low ({language_confidence:.2f}).")

    passed = len(errors) == 0

    logger.info(
        "[VALIDATION]\n"
        "  Timestamp validation: %s\n"
        "  Speaker validation: %s\n"
        "  Language validation: %s\n"
        "  Errors: %d | Warnings: %d",
        details["timestamp_validation"],
        details["speaker_validation"],
        details["language_validation"],
        len(errors),
        len(warnings),
    )

    return ValidationReport(passed=passed, warnings=warnings, errors=errors, details=details)
