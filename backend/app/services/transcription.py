"""Multilingual, timestamp-preserving faster-whisper transcription."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from typing import Any

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.services.language_detection import LanguageDetection, detect_language, detect_segment_languages
from app.utils.audio import get_audio_duration

logger = logging.getLogger(__name__)
_whisper_model: Any | None = None
_whisper_model_name: str | None = None
_whisper_error: str | None = None
_REPEATED_CHAR_RE = re.compile(r"(.)\1{8,}")


@dataclass(frozen=True)
class TranscriptWord:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    language: str = "en"
    language_name: str = "English"
    words: list[TranscriptWord] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "language": self.language,
            "language_name": self.language_name,
            "text": self.text,
            "timestamp": _fmt_timestamp(self.start),
            "words": [word.to_dict() for word in self.words],
        }

    def with_speaker(self, speaker: str | None) -> "TranscriptSegment":
        return replace(self, speaker=speaker)

    @classmethod
    def from_words(
        cls,
        words: list[TranscriptWord],
        speaker: str | None,
        language: str,
    ) -> "TranscriptSegment":
        raw_joined = "".join(word.text for word in words)
        if " " in raw_joined:
            text = re.sub(r"\s+", " ", raw_joined).strip()
        else:
            text = " ".join(word.text.strip() for word in words if word.text.strip())
        lang_name = SUPPORTED_LANGUAGES.get(language, "English")
        return cls(
            start=round(words[0].start, 3),
            end=round(words[-1].end, 3),
            text=text,
            speaker=speaker,
            language=language,
            language_name=lang_name,
            words=words,
        )

    def merged_with(self, other: "TranscriptSegment") -> "TranscriptSegment":
        combined_text = f"{self.text} {other.text}".strip()
        return TranscriptSegment(
            start=min(self.start, other.start),
            end=max(self.end, other.end),
            text=combined_text,
            speaker=self.speaker,
            language=self.language,
            language_name=self.language_name,
            words=[*self.words, *other.words],
        )


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_name: str
    language_probability: float
    language_detection: LanguageDetection
    segments: list[TranscriptSegment] = field(default_factory=list)
    detected_confident: bool = True


def _fmt_timestamp(seconds: float) -> str:
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def whisper_status() -> str:
    try:
        import faster_whisper  # noqa: F401
        return "available"
    except Exception as exc:  # noqa: BLE001
        global _whisper_error
        _whisper_error = str(exc)
        return "unavailable"


def get_whisper_model() -> Any:
    global _whisper_model, _whisper_model_name, _whisper_error
    settings = get_settings()
    wanted = (settings.whisper_model or "medium").strip()
    if _whisper_model is not None and _whisper_model_name == wanted:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel

        device = settings.device.lower() if settings.device.lower() in {"cpu", "cuda"} else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logger.info("[WHISPER] Loading model=%s device=%s compute_type=%s", wanted, device, compute_type)
        _whisper_model = WhisperModel(wanted, device=device, compute_type=compute_type)
        _whisper_model_name = wanted
        return _whisper_model
    except Exception as exc:  # noqa: BLE001
        _whisper_error = str(exc)
        logger.exception("[WHISPER] Model load failure")
        raise RuntimeError(f"Speech-to-text model '{wanted}' could not be loaded: {exc}") from exc


def _clean_unicode_text(text: str) -> str:
    """Sanitize transcript text while strictly preserving Indic and Latin Unicode characters."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = _REPEATED_CHAR_RE.sub(r"\1\1\1", text)
    return text.strip()


def _extract_words(raw_segment: Any) -> list[TranscriptWord]:
    result: list[TranscriptWord] = []
    raw_words = getattr(raw_segment, "words", None) or []
    for word in raw_words:
        start = getattr(word, "start", None)
        end = getattr(word, "end", None)
        text = getattr(word, "word", "")
        if start is not None and end is not None and float(end) > float(start):
            cleaned = str(text)
            if cleaned:
                result.append(TranscriptWord(start=float(start), end=float(end), text=cleaned))
    return result


def transcribe_audio(audio_path: str) -> TranscriptionResult:
    """Transcribe audio with accurate multi-language support, preserving native script and timestamps.

    Task is always 'transcribe' (never translation).
    """
    model = get_whisper_model()
    detection = detect_language(model, audio_path)

    # Pin language only when confidence is high and recording is not code-switched.
    # For multilingual/code-switched podcasts, allow Whisper auto-detection across windows.
    pinned_language = detection.code if (detection.confident and not detection.code_switched) else None

    logger.info(
        "[TRANSCRIPTION]\n"
        "  Detected language: %s (confidence: %.3f)\n"
        "  Pinned language: %s (auto=%s)",
        detection.code,
        detection.confidence,
        pinned_language,
        pinned_language is None,
    )

    try:
        vad_params = {
            "onset": 0.40,
            "offset": 0.30,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 400,
            "speech_pad_ms": 200,
        }
        segments_iter, info = model.transcribe(
            audio_path,
            language=pinned_language,
            task="transcribe",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=True,
            without_timestamps=False,
            vad_filter=True,
            vad_parameters=vad_params,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.60,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[TRANSCRIPTION] faster-whisper transcribe failed")
        raise RuntimeError("Transcription failed. The audio file may be corrupt or unreadable.") from exc

    default_lang = str(getattr(info, "language", None) or detection.code).lower()
    if default_lang not in SUPPORTED_LANGUAGES:
        default_lang = detection.code

    audio_duration = get_audio_duration(audio_path)
    segments: list[TranscriptSegment] = []

    for raw in segments_iter:
        start = max(0.0, float(raw.start or 0.0))
        end = min(audio_duration, float(raw.end or 0.0))
        text = _clean_unicode_text(raw.text or "")

        if not text or end <= start:
            continue

        no_speech = float(getattr(raw, "no_speech_prob", 0.0) or 0.0)
        if no_speech > 0.85 and len(text) < 6:
            continue

        words = _extract_words(raw)
        segments.append(
            TranscriptSegment(
                start=round(start, 3),
                end=round(end, 3),
                text=text,
                speaker=None,
                language=default_lang,
                language_name=SUPPORTED_LANGUAGES.get(default_lang, "English"),
                words=words,
            )
        )

    if not segments:
        raise RuntimeError("Transcription produced an empty result. The audio may be silent or unclear.")

    # Segment-level language detection for code-switched or mixed language conversations
    if detection.code_switched or pinned_language is None:
        intervals = [(seg.start, seg.end) for seg in segments]
        segment_detections = detect_segment_languages(model, audio_path, intervals, detection.code)
        for seg, seg_det in zip(segments, segment_detections, strict=True):
            if seg_det.confidence >= 0.35 and seg_det.code in SUPPORTED_LANGUAGES:
                seg.language = seg_det.code
                seg.language_name = SUPPORTED_LANGUAGES[seg_det.code]

    full_text = " ".join(seg.text for seg in segments)

    logger.info(
        "[TRANSCRIPTION]\n"
        "  Whisper segments: %d\n"
        "  Word timestamps: enabled\n"
        "  Total text chars: %d",
        len(segments),
        len(full_text),
    )

    return TranscriptionResult(
        text=full_text,
        language=detection.code,
        language_name=detection.name,
        language_probability=detection.confidence,
        language_detection=detection,
        segments=segments,
        detected_confident=detection.confident,
    )
