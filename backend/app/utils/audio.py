"""Audio helpers: FFmpeg checks, duration, normalization."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioError(Exception):
    """Raised when audio processing fails."""

    def __init__(self, message: str, stage: str = "audio"):
        super().__init__(message)
        self.message = message
        self.stage = stage


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def check_ffmpeg() -> tuple[bool, str]:
    if not shutil.which("ffmpeg"):
        return False, "FFmpeg is not installed or not on PATH. Please install FFmpeg and restart the backend."
    if not shutil.which("ffprobe"):
        return False, "ffprobe is not installed or not on PATH. Please install a full FFmpeg build."
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return False, "FFmpeg is present but failed to run."
        return True, "available"
    except Exception as exc:  # noqa: BLE001
        return False, f"FFmpeg check failed: {exc}"


def probe_audio_stream(file_path: Path | str) -> dict:
    """Return basic stream/format metadata via ffprobe."""
    path = Path(file_path)
    if not path.exists():
        raise AudioError(f"Audio file not found: {path.name}", stage="audio")

    ok, msg = check_ffmpeg()
    if not ok:
        raise AudioError(msg, stage="ffmpeg")

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    if result.returncode != 0:
        raise AudioError(
            "Could not read audio metadata. The file may be corrupt or unsupported.",
            stage="audio",
        )
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "duration": float(data.get("format", {}).get("duration", 0) or 0),
        "sample_rate": int(audio_stream.get("sample_rate") or 0),
        "channels": int(audio_stream.get("channels") or 0),
        "codec_name": str(audio_stream.get("codec_name") or ""),
        "sample_fmt": str(audio_stream.get("sample_fmt") or ""),
    }


def get_audio_duration(file_path: Path | str) -> float:
    """Return duration in seconds using ffprobe."""
    meta = probe_audio_stream(file_path)
    duration = float(meta.get("duration") or 0)
    if duration <= 0:
        raise AudioError("Audio duration could not be determined or file is empty.", stage="audio")
    return duration


def is_whisper_ready_wav(file_path: Path | str) -> bool:
    """True if file is already mono 16 kHz PCM WAV suitable for Whisper."""
    try:
        meta = probe_audio_stream(file_path)
    except AudioError:
        return False
    return (
        meta.get("codec_name") in {"pcm_s16le", "pcm_s16be"}
        and meta.get("sample_rate") == 16000
        and meta.get("channels") == 1
    )


def normalize_audio(input_path: Path | str, output_path: Path | str) -> Path:
    """
    Ensure audio is mono 16 kHz PCM WAV for Whisper.

    Skips re-encoding when the source is already Whisper-ready and destination
    can simply reuse/copy that stream once.
    """
    ok, msg = check_ffmpeg()
    if not ok:
        raise AudioError(msg, stage="ffmpeg")

    src = Path(input_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    # If source is already clean Whisper format and paths differ, copy once (no resample chain).
    if is_whisper_ready_wav(src):
        if src.resolve() == dst.resolve():
            logger.info("[Audio] Already Whisper-ready (mono/16k/PCM); reusing %s", src.name)
            return src
        cmd = ["ffmpeg", "-y", "-i", str(src), "-c", "copy", str(dst)]
        logger.info("[Audio] Source already mono 16k PCM; copying without resample")
    else:
        meta = {}
        try:
            meta = probe_audio_stream(src)
        except AudioError:
            meta = {}
        logger.info(
            "[Audio] Converting to mono 16kHz PCM WAV (src sr=%s ch=%s codec=%s)",
            meta.get("sample_rate"),
            meta.get("channels"),
            meta.get("codec_name"),
        )
        # Single conversion pass: mono + 16 kHz + PCM. Light highpass reduces rumble
        # without altering speech content.
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            "-af",
            "highpass=f=80,lowpass=f=7600",
            str(dst),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        if result.returncode != 0 or not dst.exists():
            err = (result.stderr or result.stdout or "").strip()
            snippet = err[-500:] if err else "unknown FFmpeg error"
            raise AudioError(
                f"Audio conversion failed. Ensure the file is a valid MP3/WAV/M4A. Details: {snippet}",
                stage="audio",
            )
        logger.info("[Audio] Normalized output: %s", dst.name)
        return dst
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Audio normalization failed")
        raise AudioError(f"Audio normalization failed: {exc}", stage="audio") from exc


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"
