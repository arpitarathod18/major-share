"""Multi-region Whisper language detection for supported Indian languages."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.config import SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LanguageDetection:
    code: str
    name: str
    confidence: float
    alternatives: dict[str, float]
    sampled_regions: int

    @property
    def confident(self) -> bool:
        return self.confidence >= 0.50

    @property
    def code_switched(self) -> bool:
        # True if at least two supported languages have meaningful presence
        active = [p for p in self.alternatives.values() if p >= 0.20]
        return len(active) > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "confidence": round(self.confidence, 3),
            "alternatives": {key: round(value, 3) for key, value in self.alternatives.items()},
            "sampled_regions": self.sampled_regions,
            "code_switched": self.code_switched,
        }


def _extract_speech_windows(
    audio: np.ndarray,
    sample_rate: int = 16000,
    window_duration: float = 3.0,
    max_regions: int = 10,
) -> list[tuple[int, int]]:
    """Select representative, distributed speech windows across the entire recording.

    Rejects silence, background noise, and music by inspecting frame RMS energy
    and zero-crossing rate.
    """
    total_samples = len(audio)
    if total_samples <= 0:
        return []

    window_size = int(window_duration * sample_rate)
    hop_size = int(1.0 * sample_rate)

    if total_samples <= window_size:
        return [(0, total_samples)]

    candidates: list[tuple[float, int, int]] = []

    for start in range(0, total_samples - window_size + 1, hop_size):
        end = start + window_size
        chunk = audio[start:end]

        rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        # Filter out silence and ultra-quiet audio (RMS < 0.008)
        if rms < 0.008:
            continue

        # Zero crossing rate to distinguish speech from DC rumble/hum
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2.0)
        if zcr < 0.01:  # Low frequency noise / rumble
            continue

        # Speech score based on energy and spectral variation
        speech_score = rms * (1.0 + min(1.0, zcr * 10.0))
        candidates.append((speech_score, start, end))

    if not candidates:
        # Fallback: if all audio is low volume, pick uniformly spaced slices
        step = max(1, (total_samples - window_size) // max_regions)
        return [
            (i, min(total_samples, i + window_size))
            for i in range(0, total_samples - window_size + 1, step)
        ][:max_regions]

    # Distribute samples across time buckets (e.g. 8-10 time bins)
    num_bins = min(max_regions, max(1, total_samples // (window_size * 2)))
    bin_size = total_samples / float(num_bins)
    best_per_bin: dict[int, tuple[float, int, int]] = {}

    for item in candidates:
        bin_idx = min(num_bins - 1, int(item[1] / bin_size))
        if bin_idx not in best_per_bin or item[0] > best_per_bin[bin_idx][0]:
            best_per_bin[bin_idx] = item

    selected = sorted(best_per_bin.values(), key=lambda x: x[1])
    return [(start, end) for _, start, end in selected]


def _parse_probabilities(raw: Any, detected_lang: str | None, detected_prob: float) -> dict[str, float]:
    """Extract and normalize probabilities for the supported languages (en, hi, kn, te)."""
    probs: dict[str, float] = {code: 0.0 for code in SUPPORTED_LANGUAGES}

    if isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lang_code = str(item[0]).lower().strip()
                if lang_code in SUPPORTED_LANGUAGES:
                    probs[lang_code] = max(0.0, float(item[1]))
    elif isinstance(raw, dict):
        for k, v in raw.items():
            lang_code = str(k).lower().strip()
            if lang_code in SUPPORTED_LANGUAGES:
                probs[lang_code] = max(0.0, float(v))

    if detected_lang and detected_lang in SUPPORTED_LANGUAGES:
        probs[detected_lang] = max(probs.get(detected_lang, 0.0), float(detected_prob or 0.0))

    return probs


def detect_language(model: Any, audio_path: str) -> LanguageDetection:
    """Aggregate multilingual Whisper language probabilities across representative speech regions.

    Pipeline:
      Audio -> Speech Window Selection -> Whisper Language Probability ->
      Duration/Energy Weighted Aggregation -> Final Language & Confidence
    """
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(audio_path, sampling_rate=16000)
    except Exception as exc:
        logger.warning("[LANGUAGE] Could not decode audio for language detection: %s", exc)
        return LanguageDetection("en", SUPPORTED_LANGUAGES["en"], 0.0, {c: 0.0 for c in SUPPORTED_LANGUAGES}, 0)

    windows = _extract_speech_windows(audio, sample_rate=16000)
    if not windows:
        windows = [(0, min(len(audio), 16000 * 5))]

    accumulated_votes = {code: 0.0 for code in SUPPORTED_LANGUAGES}
    total_weight = 0.0

    for start, end in windows:
        chunk = audio[start:end]
        if len(chunk) < 8000:  # < 0.5s
            continue

        try:
            language, probability, all_probs = model.detect_language(
                chunk,
                language_detection_segments=1,
                language_detection_threshold=0.0,
            )
        except Exception as exc:
            logger.debug("[LANGUAGE] Skipped region %.2f-%.2f: %s", start / 16000, end / 16000, exc)
            continue

        det_code = str(language).lower().strip() if language else None
        det_prob = float(probability or 0.0)
        parsed = _parse_probabilities(all_probs, det_code, det_prob)

        # Weight this region by RMS energy and duration
        chunk_rms = float(np.sqrt(np.mean(np.square(chunk)) + 1e-12))
        duration_factor = len(chunk) / 16000.0
        weight = max(0.2, min(2.0, chunk_rms * 25.0)) * duration_factor

        for code, val in parsed.items():
            accumulated_votes[code] += weight * val
        total_weight += weight

    if total_weight <= 0:
        logger.warning("[LANGUAGE] No speech regions returned valid language probabilities.")
        return LanguageDetection("en", SUPPORTED_LANGUAGES["en"], 0.25, {c: 0.25 for c in SUPPORTED_LANGUAGES}, len(windows))

    # Normalize accumulated votes to sum to 1.0 across supported languages
    vote_sum = sum(accumulated_votes.values())
    if vote_sum > 0:
        normalized = {code: val / vote_sum for code, val in accumulated_votes.items()}
    else:
        normalized = {code: 0.25 for code in SUPPORTED_LANGUAGES}

    best_code = max(normalized, key=normalized.get)
    confidence = normalized[best_code]

    logger.info(
        "[LANGUAGE]\n"
        "  Detected: %s (%s)\n"
        "  Confidence: %.3f\n"
        "  Regions sampled: %d\n"
        "  Probabilities: %s",
        best_code,
        SUPPORTED_LANGUAGES.get(best_code, "Unknown"),
        confidence,
        len(windows),
        {k: round(v, 3) for k, v in normalized.items()},
    )

    return LanguageDetection(
        code=best_code,
        name=SUPPORTED_LANGUAGES[best_code],
        confidence=confidence,
        alternatives=normalized,
        sampled_regions=len(windows),
    )


def detect_segment_languages(
    model: Any,
    audio_path: str,
    intervals: list[tuple[float, float]],
    fallback_code: str,
) -> list[LanguageDetection]:
    """Detect spoken language per segment interval, supporting code-switching across speech turns."""
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(audio_path, sampling_rate=16000)
    except Exception as exc:
        logger.warning("[LANGUAGE] Audio decode failed for segment detection: %s", exc)
        return [
            LanguageDetection(fallback_code, SUPPORTED_LANGUAGES.get(fallback_code, "English"), 0.0, {}, 0)
            for _ in intervals
        ]

    results: list[LanguageDetection] = []
    fallback_lang = fallback_code if fallback_code in SUPPORTED_LANGUAGES else "en"

    for start_sec, end_sec in intervals:
        start_idx = max(0, int(start_sec * 16000))
        end_idx = min(len(audio), int(end_sec * 16000))
        clip = audio[start_idx:end_idx]

        if len(clip) < 8000:  # < 0.5s
            results.append(
                LanguageDetection(fallback_lang, SUPPORTED_LANGUAGES[fallback_lang], 0.0, {}, 0)
            )
            continue

        try:
            language, probability, all_probs = model.detect_language(
                clip,
                language_detection_segments=1,
                language_detection_threshold=0.0,
            )
            det_code = str(language).lower().strip() if language else None
            det_prob = float(probability or 0.0)
            parsed = _parse_probabilities(all_probs, det_code, det_prob)
            total = sum(parsed.values())
            if total > 0:
                normalized = {k: v / total for k, v in parsed.items()}
                best = max(normalized, key=normalized.get)
                conf = normalized[best]
                results.append(
                    LanguageDetection(best, SUPPORTED_LANGUAGES[best], conf, normalized, 1)
                )
            else:
                results.append(
                    LanguageDetection(fallback_lang, SUPPORTED_LANGUAGES[fallback_lang], 0.0, {}, 1)
                )
        except Exception:
            results.append(
                LanguageDetection(fallback_lang, SUPPORTED_LANGUAGES[fallback_lang], 0.0, {}, 0)
            )

    return results
