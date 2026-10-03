"""Tests for PyAnnote Community-1 speaker diarization integration, alignment, and formatting."""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

from app.services.alignment import align_transcript
from app.services.diarization import (
    COMMUNITY_MODEL_ID,
    DiarizationResult,
    SpeakerTurn,
    _apply_runtime_shims,
    _stable_turns,
    diarization_status,
    diarize_audio,
)
from app.services.jobs import format_transcript_text
from app.services.transcription import TranscriptSegment, TranscriptWord


def test_runtime_shims_and_imports():
    """Ensure runtime compatibility shims execute and pyannote imports cleanly."""
    _apply_runtime_shims()
    import torchaudio
    import torch
    import huggingface_hub

    assert hasattr(torchaudio, "AudioMetaData")
    assert hasattr(torchaudio, "list_audio_backends")
    assert callable(huggingface_hub.hf_hub_download)
    assert callable(huggingface_hub.snapshot_download)


def test_diarization_status_structure():
    """Verify diarization_status returns rich metadata dictionary."""
    status = diarization_status()
    assert isinstance(status, dict)
    assert "available" in status
    assert "provider" in status
    assert status["provider"] == "pyannote"
    assert "model" in status
    assert "reason" in status


def test_single_speaker_formatting_omits_speaker_label():
    """Verify single-speaker recordings do NOT display 'SPEAKER_00' in transcript lines."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=5.42,
            text="ನಮಸ್ಕಾರ, ಎಲ್ಲರಿಗೂ ಸ್ವಾಗತ.",
            language="kn",
            language_name="Kannada",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=5.42,
            end=10.83,
            text="ಇಂದು ನಾವು ಮುಖ್ಯ ವಿಷಯವನ್ನು ಚರ್ಚಿಸುತ್ತೇವೆ.",
            language="kn",
            language_name="Kannada",
            speaker="SPEAKER_00",
        ),
    ]

    # For single speaker (detected_speakers = 1)
    formatted_single = format_transcript_text(segments, detected_speakers=1)
    assert "SPEAKER_00" not in formatted_single
    assert "00:00.000 → 00:05.420 | Kannada\nನಮಸ್ಕಾರ, ಎಲ್ಲರಿಗೂ ಸ್ವಾಗತ." in formatted_single
    assert "00:05.420 → 00:10.830 | Kannada\nಇಂದು ನಾವು ಮುಖ್ಯ ವಿಷಯವನ್ನು ಚರ್ಚಿಸುತ್ತೇವೆ." in formatted_single


def test_multi_speaker_formatting_includes_speaker_labels():
    """Verify multi-speaker recordings display stable SPEAKER_XX labels."""
    segments = [
        TranscriptSegment(
            start=0.0,
            end=5.42,
            text="Welcome to the podcast.",
            language="en",
            language_name="English",
            speaker="SPEAKER_00",
        ),
        TranscriptSegment(
            start=5.42,
            end=11.40,
            text="Thank you, happy to be here.",
            language="en",
            language_name="English",
            speaker="SPEAKER_01",
        ),
    ]

    formatted_multi = format_transcript_text(segments, detected_speakers=2)
    assert "SPEAKER_00 | 00:00.000 → 00:05.420 | English\nWelcome to the podcast." in formatted_multi
    assert "SPEAKER_01 | 00:05.420 → 00:11.400 | English\nThank you, happy to be here." in formatted_multi


def test_no_fake_speaker_labels_on_fallback():
    """Verify that if diarization fails/is unavailable, speaker is None and no fake labels are created."""
    segments = [
        TranscriptSegment(start=0.0, end=4.0, text="First sentence.", language="en"),
        TranscriptSegment(start=4.0, end=8.0, text="Second sentence.", language="en"),
        TranscriptSegment(start=8.0, end=12.0, text="Third sentence.", language="en"),
    ]
    # No turns from diarization
    aligned = align_transcript(segments, turns=[])
    assert len(aligned) == 3
    assert all(seg.speaker is None for seg in aligned)

    # Formatted transcript without diarization (detected_speakers = 0)
    formatted = format_transcript_text(aligned, detected_speakers=0)
    assert "SPEAKER_00" not in formatted
    assert "SPEAKER_01" not in formatted


def test_stable_turns_chronological_ordering_and_smoothing():
    """Verify raw turns are sorted, smoothed, and assigned stable sequential IDs."""
    raw = [
        (6.0, 10.0, "SPEAKER_B"),
        (0.2, 5.0, "SPEAKER_A"),
        (5.1, 5.9, "SPEAKER_A"),  # Gap <= 0.25s, should merge with first turn
        (10.5, 10.6, "NOISE"),    # < 0.20s fragment, should be filtered
    ]
    turns = _stable_turns(raw)
    assert len(turns) == 2
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].start == 0.2
    assert turns[0].end == 5.9
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[1].start == 6.0
    assert turns[1].end == 10.0


def test_diarization_real_audio_pipeline():
    """Test diarization on a synthetic 16kHz WAV audio file."""
    # Create 3-second 16kHz test WAV audio
    sample_rate = 16000
    duration_s = 3.0
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    # Generate dual-frequency tone
    audio_data = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_wav_path = f.name
        with wave.open(temp_wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())

    try:
        result = diarize_audio(temp_wav_path)
        assert isinstance(result, DiarizationResult)
        assert isinstance(result.available, bool)
        assert isinstance(result.speaker_count, int)
        assert isinstance(result.turns, list)
    finally:
        Path(temp_wav_path).unlink(missing_ok=True)
