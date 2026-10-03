"""Tests for speaker-label stability and time-based alignment."""

from app.services.alignment import align_transcript
from app.services.diarization import SpeakerTurn, _stable_turns
from app.services.transcription import TranscriptSegment, TranscriptWord


def test_stable_turns_use_first_appearance_and_merge_fragments():
    turns = _stable_turns([(2, 3, "B"), (0, 1, "A"), (1.1, 1.8, "A"), (3, 4, "B")])
    assert [(turn.speaker, turn.start, turn.end) for turn in turns] == [
        ("SPEAKER_00", 0, 1.8), ("SPEAKER_01", 2, 4)
    ]


def test_alignment_splits_one_whisper_segment_at_speaker_turn():
    source = TranscriptSegment(0, 4, "hello there fine thanks", language="en", words=[
        TranscriptWord(0, 1, " hello"), TranscriptWord(1, 2, " there"),
        TranscriptWord(2, 3, " fine"), TranscriptWord(3, 4, " thanks"),
    ])
    result = align_transcript(source and [source], [SpeakerTurn(0, 2, "SPEAKER_00"), SpeakerTurn(2, 4, "SPEAKER_01")])
    assert [(item.speaker, item.start, item.end) for item in result] == [
        ("SPEAKER_00", 0, 2), ("SPEAKER_01", 2, 4)
    ]
