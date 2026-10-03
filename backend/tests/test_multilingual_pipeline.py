"""Comprehensive test suite covering all multilingual podcast scenarios:
- Test 1: English - 2 speakers
- Test 2: Hindi - 2 speakers
- Test 3: Kannada - 2 speakers
- Test 4: Telugu - 2 speakers
- Test 5: Kannada + English - 2 speakers (code-switched)
- Test 6: Telugu + English - 2 speakers (code-switched)
- Test 7: Hindi + English - 2 speakers (code-switched)
- Test 8: 3-speaker conversation
- Test 9: One-speaker recording
- Test 10: Long conversation with multiple speaker changes
- Test 11: Background noise & speech window selection
- Test 12: Speaker changes language during conversation (preserving same speaker ID)
"""

from __future__ import annotations

import numpy as np

from app.services.alignment import align_transcript
from app.services.diarization import SpeakerTurn, _stable_turns
from app.services.language_detection import (
    LanguageDetection,
    _extract_speech_windows,
    _parse_probabilities,
)
from app.services.transcription import TranscriptSegment, TranscriptWord, _clean_unicode_text
from app.services.validation import validate_results


def test_1_english_2_speakers():
    """Verify 2-speaker English conversation alignment and timestamps."""
    words_spk0 = [
        TranscriptWord(0.0, 1.2, "Welcome"),
        TranscriptWord(1.2, 2.0, "everyone"),
        TranscriptWord(2.0, 3.5, "to our show."),
    ]
    words_spk1 = [
        TranscriptWord(3.6, 4.8, "Thank you,"),
        TranscriptWord(4.8, 6.0, "great to be here."),
    ]
    raw_segments = [
        TranscriptSegment(0.0, 3.5, "Welcome everyone to our show.", language="en", language_name="English", words=words_spk0),
        TranscriptSegment(3.6, 6.0, "Thank you, great to be here.", language="en", language_name="English", words=words_spk1),
    ]
    turns = [
        SpeakerTurn(0.0, 3.5, "SPEAKER_00"),
        SpeakerTurn(3.6, 6.0, "SPEAKER_01"),
    ]
    aligned = align_transcript(raw_segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "en"
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].language == "en"


def test_2_hindi_2_speakers_unicode():
    """Verify Hindi 2-speaker conversation and Devanagari Unicode preservation."""
    hindi_text_0 = "नमस्ते, आज हम भारतीय पॉडकास्ट पर चर्चा कर रहे हैं।"
    hindi_text_1 = "हाँ, यह बहुत महत्वपूर्ण विषय है।"
    assert _clean_unicode_text(hindi_text_0) == hindi_text_0

    words_0 = [
        TranscriptWord(0.0, 2.0, "नमस्ते,"),
        TranscriptWord(2.0, 5.0, "आज हम भारतीय पॉडकास्ट पर चर्चा कर रहे हैं।"),
    ]
    words_1 = [
        TranscriptWord(5.2, 8.0, "हाँ, यह बहुत महत्वपूर्ण विषय है।"),
    ]
    raw_segments = [
        TranscriptSegment(0.0, 5.0, hindi_text_0, language="hi", language_name="Hindi", words=words_0),
        TranscriptSegment(5.2, 8.0, hindi_text_1, language="hi", language_name="Hindi", words=words_1),
    ]
    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(5.2, 8.0, "SPEAKER_01"),
    ]
    aligned = align_transcript(raw_segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "hi"
    assert "नमस्ते" in aligned[0].text
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].language == "hi"
    assert "महत्वपूर्ण" in aligned[1].text


def test_3_kannada_2_speakers_unicode():
    """Verify Kannada 2-speaker conversation and Kannada Unicode preservation."""
    kn_text_0 = "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ, ನಮ್ಮ ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗೆ ಸ್ವಾಗತ."
    kn_text_1 = "ಧನ್ಯವಾದಗಳು, ಈ ವಿಷಯದ ಬಗ್ಗೆ ಮಾತನಾಡಲು ಸಂತೋಷವಾಗಿದೆ."
    assert _clean_unicode_text(kn_text_0) == kn_text_0

    words_0 = [
        TranscriptWord(0.0, 2.5, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ,"),
        TranscriptWord(2.5, 5.4, "ನಮ್ಮ ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗೆ ಸ್ವಾಗತ."),
    ]
    words_1 = [
        TranscriptWord(5.5, 9.8, "ಧನ್ಯವಾದಗಳು, ಈ ವಿಷಯದ ಬಗ್ಗೆ ಮಾತನಾಡಲು ಸಂತೋಷವಾಗಿದೆ."),
    ]
    raw_segments = [
        TranscriptSegment(0.0, 5.4, kn_text_0, language="kn", language_name="Kannada", words=words_0),
        TranscriptSegment(5.5, 9.8, kn_text_1, language="kn", language_name="Kannada", words=words_1),
    ]
    turns = [
        SpeakerTurn(0.0, 5.4, "SPEAKER_00"),
        SpeakerTurn(5.5, 9.8, "SPEAKER_01"),
    ]
    aligned = align_transcript(raw_segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "kn"
    assert "ನಮಸ್ಕಾರ" in aligned[0].text
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].language == "kn"
    assert "ಧನ್ಯವಾದಗಳು" in aligned[1].text


def test_4_telugu_2_speakers_unicode():
    """Verify Telugu 2-speaker conversation and Telugu Unicode preservation."""
    te_text_0 = "నమస్కారం అందరికీ, ఈ రోజు మన ముఖ్యమైన అంశం గురించి మాట్లాడుకుందాం."
    te_text_1 = "ఖచ్చితంగా, ఇది అందరికీ ఉపయోగపడుతుంది."
    assert _clean_unicode_text(te_text_0) == te_text_0

    words_0 = [
        TranscriptWord(0.0, 4.8, "నమస్కారం అందరికీ, ఈ రోజు మన ముఖ్యమైన అంశం గురించి మాట్లాడుకుందాం."),
    ]
    words_1 = [
        TranscriptWord(4.9, 9.4, "ఖచ్చితంగా, ఇది అందరికీ ఉపయోగపడుతుంది."),
    ]
    raw_segments = [
        TranscriptSegment(0.0, 4.8, te_text_0, language="te", language_name="Telugu", words=words_0),
        TranscriptSegment(4.9, 9.4, te_text_1, language="te", language_name="Telugu", words=words_1),
    ]
    turns = [
        SpeakerTurn(0.0, 4.8, "SPEAKER_00"),
        SpeakerTurn(4.9, 9.4, "SPEAKER_01"),
    ]
    aligned = align_transcript(raw_segments, turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "te"
    assert "నమస్కారం" in aligned[0].text
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].language == "te"
    assert "ఉపయోగపడుతుంది" in aligned[1].text


def test_5_kannada_plus_english_code_switching():
    """Verify Kannada + English code-switching conversation."""
    seg1 = TranscriptSegment(0.0, 4.0, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ", language="kn", language_name="Kannada", words=[TranscriptWord(0.0, 4.0, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ")])
    seg2 = TranscriptSegment(4.2, 8.0, "Welcome to the tech edition.", language="en", language_name="English", words=[TranscriptWord(4.2, 8.0, "Welcome to the tech edition.")])
    turns = [SpeakerTurn(0.0, 4.0, "SPEAKER_00"), SpeakerTurn(4.2, 8.0, "SPEAKER_01")]
    aligned = align_transcript([seg1, seg2], turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "kn"
    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].language == "en"


def test_6_telugu_plus_english_code_switching():
    """Verify Telugu + English code-switching conversation."""
    seg1 = TranscriptSegment(0.0, 3.5, "నమస్కారం", language="te", language_name="Telugu", words=[TranscriptWord(0.0, 3.5, "నమస్కారం")])
    seg2 = TranscriptSegment(3.6, 7.0, "Today we discuss artificial intelligence.", language="en", language_name="English", words=[TranscriptWord(3.6, 7.0, "Today we discuss artificial intelligence.")])
    turns = [SpeakerTurn(0.0, 3.5, "SPEAKER_00"), SpeakerTurn(3.6, 7.0, "SPEAKER_01")]
    aligned = align_transcript([seg1, seg2], turns)
    assert len(aligned) == 2
    assert aligned[0].language == "te"
    assert aligned[1].language == "en"


def test_7_hindi_plus_english_code_switching():
    """Verify Hindi + English code-switching conversation."""
    seg1 = TranscriptSegment(0.0, 3.0, "नमस्ते दोस्तों", language="hi", language_name="Hindi", words=[TranscriptWord(0.0, 3.0, "नमस्ते दोस्तों")])
    seg2 = TranscriptSegment(3.2, 6.0, "Let us get right into the topic.", language="en", language_name="English", words=[TranscriptWord(3.2, 6.0, "Let us get right into the topic.")])
    turns = [SpeakerTurn(0.0, 3.0, "SPEAKER_00"), SpeakerTurn(3.2, 6.0, "SPEAKER_01")]
    aligned = align_transcript([seg1, seg2], turns)
    assert len(aligned) == 2
    assert aligned[0].language == "hi"
    assert aligned[1].language == "en"


def test_8_three_speaker_conversation():
    """Verify 3-speaker conversation diarization and alignment."""
    raw_turns = [
        (0.0, 4.0, "HOST"),
        (4.2, 8.0, "GUEST_1"),
        (8.2, 12.0, "GUEST_2"),
        (12.2, 16.0, "HOST"),
    ]
    turns = _stable_turns(raw_turns)
    speakers = {turn.speaker for turn in turns}
    assert len(speakers) == 3
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[2].speaker == "SPEAKER_02"
    assert turns[3].speaker == "SPEAKER_00"  # Host returns as SPEAKER_00


def test_9_one_speaker_recording():
    """Verify monologue with one speaker."""
    raw_turns = [
        (0.0, 5.0, "SOLO"),
        (5.5, 12.0, "SOLO"),
        (12.5, 20.0, "SOLO"),
    ]
    turns = _stable_turns(raw_turns)
    speakers = {turn.speaker for turn in turns}
    assert len(speakers) == 1
    assert turns[0].speaker == "SPEAKER_00"


def test_10_long_conversation_multiple_speaker_turns():
    """Verify long conversation with rapid speaker switching and gap merging."""
    raw_turns = [
        (0.0, 2.0, "A"),
        (2.1, 4.0, "A"),  # Should merge with turn 1 (gap <= 0.22)
        (4.5, 7.0, "B"),
        (7.5, 10.0, "A"),
        (10.5, 13.0, "B"),
        (13.5, 16.0, "A"),
    ]
    turns = _stable_turns(raw_turns)
    assert len(turns) == 5
    assert turns[0].start == 0.0 and turns[0].end == 4.0
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[1].speaker == "SPEAKER_01"
    assert turns[2].speaker == "SPEAKER_00"
    assert turns[3].speaker == "SPEAKER_01"
    assert turns[4].speaker == "SPEAKER_00"


def test_11_speech_window_extraction_rejects_silence_and_noise():
    """Verify that speech window extraction rejects low-energy silent chunks and samples distributed regions."""
    rate = 16000
    # 10 seconds of silence + 5 seconds of active speech + 10 seconds of silence
    silence = np.zeros(10 * rate, dtype=np.float32)
    # Active tone with amplitude 0.2
    t = np.linspace(0, 5, 5 * rate, endpoint=False)
    speech = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    audio = np.concatenate([silence, speech, silence])

    windows = _extract_speech_windows(audio, sample_rate=rate, window_duration=2.0)
    assert len(windows) > 0
    # Verified that sampled window is in the active speech interval (10s to 15s)
    found_in_active_region = any(start >= 9 * rate and end <= 16 * rate for start, end in windows)
    assert found_in_active_region


def test_12_speaker_changes_language_preserves_speaker_id():
    """Verify that when a single speaker changes language (Kannada -> English -> Kannada),

    their speaker ID remains strictly stable.
    """
    raw_segments = [
        TranscriptSegment(0.0, 5.0, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ", language="kn", language_name="Kannada", words=[TranscriptWord(0.0, 5.0, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ")]),
        TranscriptSegment(5.2, 9.0, "Let us switch to English for a moment.", language="en", language_name="English", words=[TranscriptWord(5.2, 9.0, "Let us switch to English for a moment.")]),
        TranscriptSegment(9.2, 14.0, "ಮತ್ತೆ ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸೋಣ.", language="kn", language_name="Kannada", words=[TranscriptWord(9.2, 14.0, "ಮತ್ತೆ ಕನ್ನಡದಲ್ಲಿ ಮುಂದುವರಿಸೋಣ.")]),
    ]
    # Single person speaking throughout
    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(5.2, 9.0, "SPEAKER_00"),
        SpeakerTurn(9.2, 14.0, "SPEAKER_00"),
    ]
    aligned = align_transcript(raw_segments, turns)
    assert len(aligned) == 3
    # Same speaker throughout
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].language == "kn"
    assert aligned[1].speaker == "SPEAKER_00"
    assert aligned[1].language == "en"
    assert aligned[2].speaker == "SPEAKER_00"
    assert aligned[2].language == "kn"


def test_word_level_boundary_split_inside_segment():
    """Verify splitting a single Whisper segment containing multiple speaker turns."""
    words = [
        TranscriptWord(10.0, 11.0, "Hello"),
        TranscriptWord(11.0, 12.0, "how"),
        TranscriptWord(12.0, 13.0, "are"),
        TranscriptWord(13.0, 14.0, "you?"),
        TranscriptWord(14.0, 16.0, "I"),
        TranscriptWord(16.0, 18.0, "am"),
        TranscriptWord(18.0, 20.0, "fine."),
    ]
    seg = TranscriptSegment(10.0, 20.0, "Hello how are you? I am fine.", language="en", language_name="English", words=words)
    turns = [
        SpeakerTurn(10.0, 14.0, "SPEAKER_00"),
        SpeakerTurn(14.0, 20.0, "SPEAKER_01"),
    ]
    aligned = align_transcript([seg], turns)
    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].start == 10.0
    assert aligned[0].end == 14.0
    assert "Hello how are you?" in aligned[0].text

    assert aligned[1].speaker == "SPEAKER_01"
    assert aligned[1].start == 14.0
    assert aligned[1].end == 20.0
    assert "I am fine." in aligned[1].text


def test_validation_quality_report():
    """Verify quality validation checks pass for valid aligned segments."""
    segments = [
        TranscriptSegment(0.0, 4.0, "ನಮಸ್ಕಾರ", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
        TranscriptSegment(4.2, 8.0, "ಧನ್ಯವಾದಗಳು", speaker="SPEAKER_01", language="kn", language_name="Kannada"),
    ]
    report = validate_results(
        audio_duration=8.5,
        detected_language="kn",
        language_confidence=0.92,
        segments=segments,
        diarization_available=True,
        speakers=["SPEAKER_00", "SPEAKER_01"],
    )
    assert report.passed is True
    assert report.details["timestamp_validation"] == "PASS"
    assert "PASS" in report.details["speaker_validation"]
    assert report.details["language_validation"] == "PASS"
    assert len(report.errors) == 0


def test_single_speaker_formatting_hides_speaker_labels():
    """Verify that single-speaker recording formats with timestamps and language but NO speaker labels."""
    from app.services.jobs import format_transcript_text

    segments = [
        TranscriptSegment(0.0, 5.42, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ ನಮ್ಮ ಕಾರ್ಯಕ್ರಮಕ್ಕೆ ಸ್ವಾಗತ.", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
        TranscriptSegment(5.42, 10.83, "ಇವತ್ತು ನಾವು ಪ್ರಮುಖ ವಿಷಯಗಳ ಬಗ್ಗೆ ಮಾತನಾಡಲಿದ್ದೇವೆ.", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
        TranscriptSegment(10.83, 16.24, "ದಯವಿಟ್ಟು ಗಮನವಿಟ್ಟು ಕೇಳಿ.", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
    ]

    formatted = format_transcript_text(segments, detected_speakers=1)

    # Must contain timestamps
    assert "00:00.000 → 00:05.420" in formatted
    assert "00:05.420 → 00:10.830" in formatted
    assert "00:10.830 → 00:16.240" in formatted
    assert "Kannada" in formatted

    # Must NOT contain speaker labels
    assert "SPEAKER_00" not in formatted
    assert "Speaker 1" not in formatted
    assert "SPEAKER_" not in formatted


def test_single_speaker_multilingual_code_switching_remains_single_speaker():
    """Verify a single person speaking Kannada -> English -> Telugu -> Hindi.

    The result remains a 1-speaker transcript with language tags and timestamps,
    and NO artificial speaker labels.
    """
    from app.services.jobs import format_transcript_text

    segments = [
        TranscriptSegment(0.0, 5.0, "ನಮಸ್ಕಾರ ಸ್ನೇಹಿತರೆ", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
        TranscriptSegment(5.0, 10.0, "Welcome to this multilingual session", speaker="SPEAKER_00", language="en", language_name="English"),
        TranscriptSegment(10.0, 15.0, "నమస్కారం అందరికీ", speaker="SPEAKER_00", language="te", language_name="Telugu"),
        TranscriptSegment(15.0, 20.0, "नमस्ते आप सभी का स्वागत है", speaker="SPEAKER_00", language="hi", language_name="Hindi"),
    ]

    turns = [
        SpeakerTurn(0.0, 5.0, "SPEAKER_00"),
        SpeakerTurn(5.0, 10.0, "SPEAKER_00"),
        SpeakerTurn(10.0, 15.0, "SPEAKER_00"),
        SpeakerTurn(15.0, 20.0, "SPEAKER_00"),
    ]

    aligned = align_transcript(segments, turns)
    speakers_list = sorted({seg.speaker for seg in aligned if seg.speaker})
    detected_speakers = len(speakers_list)

    assert detected_speakers == 1

    formatted = format_transcript_text(aligned, detected_speakers=detected_speakers)

    # Must retain timestamps and each language
    assert "00:00.000 → 00:05.000 | Kannada" in formatted
    assert "00:05.000 → 00:10.000 | English" in formatted
    assert "00:10.000 → 00:15.000 | Telugu" in formatted
    assert "00:15.000 → 00:20.000 | Hindi" in formatted

    # Must NOT create speaker labels just because language changed
    assert "SPEAKER_00" not in formatted
    assert "SPEAKER_01" not in formatted
    assert "SPEAKER_02" not in formatted
    assert "SPEAKER_03" not in formatted


def test_multi_speaker_formatting_displays_speaker_labels():
    """Verify that multi-speaker audio (>=2 speakers) formats with stable speaker labels."""
    from app.services.jobs import format_transcript_text

    segments = [
        TranscriptSegment(0.0, 5.42, "ನಮಸ್ಕಾರ ಎಲ್ಲರಿಗೂ", speaker="SPEAKER_00", language="kn", language_name="Kannada"),
        TranscriptSegment(5.42, 9.87, "ಧನ್ಯವಾದಗಳು, ಶುಭೋದಯ", speaker="SPEAKER_01", language="kn", language_name="Kannada"),
        TranscriptSegment(9.87, 14.21, "How are you today?", speaker="SPEAKER_00", language="en", language_name="English"),
    ]

    formatted = format_transcript_text(segments, detected_speakers=2)

    # Must contain speaker labels and timestamps
    assert "SPEAKER_00 | 00:00.000 → 00:05.420 | Kannada" in formatted
    assert "SPEAKER_01 | 00:05.420 → 00:09.870 | Kannada" in formatted
    assert "SPEAKER_00 | 00:09.870 → 00:14.210 | English" in formatted


def test_audio_summary_pipeline_preserved():
    """Verify that summarization and TTS pipeline remain active for Indian languages."""
    from app.services.summarization import summarize_transcript
    from app.services.tts import language_supported_for_tts, tts_supported_languages

    text = "This is an important discussion about technology and artificial intelligence in modern India."
    summary, method, warning, lang, err = summarize_transcript(text, output_language="en", source_language="en")

    assert summary is not None and len(summary) > 0
    assert lang == "en"

    # Verify TTS languages
    for lang_code in ("en", "hi", "kn", "te"):
        assert language_supported_for_tts(lang_code) is True

    supported = tts_supported_languages()
    assert set(supported) == {"en", "hi", "kn", "te"}


