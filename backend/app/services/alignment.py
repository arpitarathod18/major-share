"""Temporal alignment between Whisper words and voice-diarization turns."""

from __future__ import annotations

import logging
from app.services.diarization import SpeakerTurn
from app.services.transcription import TranscriptSegment, TranscriptWord

logger = logging.getLogger(__name__)


def _speaker_for_interval(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    """Calculate maximum temporal overlap with diarization speaker turns."""
    if not turns or end <= start:
        return None

    best_overlap = 0.0
    best_speaker: str | None = None

    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.speaker

    # Require at least minimal overlap
    if best_overlap > 0.05:
        return best_speaker
    return None


def align_transcript(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
) -> list[TranscriptSegment]:
    """Align Whisper segments with diarization turns using word-level timestamps.

    If two speakers speak within a single Whisper segment, word timestamps split
    the segment at the exact speaker transition boundary.
    """
    if not turns:
        # Diarization unavailable or no turns found; preserve original segments with speaker=None
        return [seg.with_speaker(None) for seg in segments]

    aligned: list[TranscriptSegment] = []

    for seg in segments:
        if not seg.words:
            # Segment has no individual word timestamps; match entire interval
            speaker = _speaker_for_interval(seg.start, seg.end, turns)
            aligned.append(seg.with_speaker(speaker))
            continue

        # Group words into contiguous same-speaker chunks
        current_words: list[TranscriptWord] = []
        current_speaker: str | None = None

        for word in seg.words:
            word_speaker = _speaker_for_interval(word.start, word.end, turns)
            if not current_words:
                current_words.append(word)
                current_speaker = word_speaker
            elif word_speaker == current_speaker:
                current_words.append(word)
            else:
                # Speaker switch inside Whisper segment! Emit current chunk
                aligned.append(
                    TranscriptSegment.from_words(
                        words=current_words,
                        speaker=current_speaker,
                        language=seg.language,
                    )
                )
                current_words = [word]
                current_speaker = word_speaker

        if current_words:
            aligned.append(
                TranscriptSegment.from_words(
                    words=current_words,
                    speaker=current_speaker,
                    language=seg.language,
                )
            )

    # Merge adjacent segments belonging to the same speaker and same language
    merged: list[TranscriptSegment] = []
    for seg in aligned:
        if (
            merged
            and merged[-1].speaker == seg.speaker
            and merged[-1].language == seg.language
            and seg.start - merged[-1].end <= 0.30
        ):
            merged[-1] = merged[-1].merged_with(seg)
        else:
            merged.append(seg)

    logger.info(
        "[ALIGNMENT]\n"
        "  Whisper source segments: %d\n"
        "  Diarization turns: %d\n"
        "  Final aligned segments: %d",
        len(segments),
        len(turns),
        len(merged),
    )

    return merged
