"""In-memory job store and podcast processing pipeline."""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.services.alignment import align_transcript
from app.services.diarization import diarize_audio
from app.services.summarization import summarize_transcript
from app.services.transcription import transcribe_audio
from app.services.translation import translate_text
from app.services.tts import language_supported_for_tts, synthesize_speech
from app.services.validation import validate_results
from app.utils.audio import AudioError, format_timestamp, get_audio_duration, normalize_audio
from app.utils.files import cleanup_temp_files, write_text

logger = logging.getLogger(__name__)

STAGES = [
    ("upload", "Uploading audio", 5),
    ("language_detection", "Detecting language", 15),
    ("transcription", "Transcribing audio", 40),
    ("diarization", "Identifying speakers", 60),
    ("summarization", "Generating summary", 80),
    ("tts", "Generating audio summary", 95),
    ("completed", "Completed", 100),
]

STAGE_PROGRESS = {s[0]: s[2] for s in STAGES}
STAGE_LABELS = {s[0]: s[1] for s in STAGES}


def _fmt_ts_precise(seconds: float) -> str:
    """Format seconds into MM:SS.mmm (or HH:MM:SS.mmm for >=1hr)."""
    total_ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    return f"{m:02d}:{s:02d}.{ms:03d}"


def format_transcript_text(segments: list[Any], detected_speakers: int) -> str:
    """Format transcript segments into standard readable lines.

    For Single Speaker (detected_speakers <= 1):
      00:00.000 → 00:05.420 | Kannada
      Transcript...

    For Multiple Speakers (detected_speakers >= 2):
      SPEAKER_00 | 00:00.000 → 00:05.420 | Kannada
      Transcript...
    """
    blocks: list[str] = []
    is_multi_speaker = detected_speakers >= 2

    for seg in segments:
        start_val = getattr(seg, "start", 0.0) if not isinstance(seg, dict) else seg.get("start", 0.0)
        end_val = getattr(seg, "end", 0.0) if not isinstance(seg, dict) else seg.get("end", 0.0)
        lang_name = getattr(seg, "language_name", "") if not isinstance(seg, dict) else seg.get("language_name", "")
        text = (getattr(seg, "text", "") if not isinstance(seg, dict) else seg.get("text", "")).strip()
        speaker = getattr(seg, "speaker", None) if not isinstance(seg, dict) else seg.get("speaker")

        ts_start = _fmt_ts_precise(float(start_val))
        ts_end = _fmt_ts_precise(float(end_val))

        header = f"{ts_start} → {ts_end}"
        if lang_name:
            header = f"{header} | {lang_name}"

        if is_multi_speaker and speaker:
            header = f"{speaker} | {header}"

        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


@dataclass
class Job:
    job_id: str
    status: str = "queued"
    stage: str = "upload"
    progress: int = 0
    stage_label: str = STAGE_LABELS["upload"]
    error: str | None = None
    error_stage: str | None = None
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    upload_id: str | None = None
    original_filename: str | None = None
    stored_path: str | None = None
    output_language: str = "en"
    result: dict[str, Any] | None = None

    def to_status(self) -> dict[str, Any]:
        elapsed = None
        if self.started_at:
            end = self.finished_at or time.time()
            elapsed = round(end - self.started_at, 2)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": self.progress,
            "elapsed_seconds": elapsed,
            "error": self.error,
            "error_stage": self.error_stage,
            "warnings": self.warnings,
        }


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        upload_id: str,
        stored_path: str,
        original_filename: str,
        output_language: str,
    ) -> Job:
        job_id = uuid.uuid4().hex[:12]
        job = Job(
            job_id=job_id,
            status="processing",
            stage="upload",
            progress=5,
            upload_id=upload_id,
            stored_path=stored_path,
            original_filename=original_filename,
            output_language=output_language if output_language in SUPPORTED_LANGUAGES else "en",
            started_at=time.time(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            for k, v in kwargs.items():
                setattr(job, k, v)
            if "stage" in kwargs:
                job.stage_label = STAGE_LABELS.get(job.stage, job.stage)
                if "progress" not in kwargs:
                    job.progress = STAGE_PROGRESS.get(job.stage, job.progress)


job_manager = JobManager()


def start_processing(job_id: str) -> None:
    thread = threading.Thread(target=_run_pipeline, args=(job_id,), daemon=True)
    thread.start()


def _run_pipeline(job_id: str) -> None:
    settings = get_settings()
    job = job_manager.get(job_id)
    if not job or not job.stored_path:
        return

    logger.info("[INFO] Job %s started", job_id)
    temp_wav: Path | None = None
    warnings: list[str] = []

    try:
        source = Path(job.stored_path)
        if not source.exists():
            raise RuntimeError("Uploaded audio file is missing. Please upload again.")

        # 1. Validate audio duration
        job_manager.update(job_id, stage="upload", progress=8)
        try:
            duration = get_audio_duration(source)
        except AudioError as exc:
            raise RuntimeError(exc.message) from exc

        if duration > settings.max_audio_duration:
            minutes = settings.max_audio_duration // 60
            raise RuntimeError(
                f"Audio is too long ({duration / 60:.1f} min). Maximum recommended duration is {minutes} minutes."
            )

        # 2. Normalize audio to 16kHz mono WAV
        temp_wav = settings.temp_dir / f"{job_id}_16k.wav"
        job_manager.update(job_id, stage="language_detection", progress=12)
        try:
            normalize_audio(source, temp_wav)
        except AudioError as exc:
            raise RuntimeError(exc.message) from exc

        # 3. Detect language + Transcribe (with word timestamps)
        job_manager.update(job_id, stage="transcription", progress=20)
        logger.info("[INFO] Transcription & language detection started")
        try:
            transcription = transcribe_audio(str(temp_wav))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Transcription error for job %s", job_id)
            raise RuntimeError("Transcription failed. Please try another audio file or a shorter clip.") from exc
        logger.info("[INFO] Transcription completed")

        if not transcription.detected_confident:
            warnings.append("Language detection confidence was low. Spoken language may be mixed or approximate.")

        # 4. Speaker Diarization + Word-Level Alignment
        job_manager.update(job_id, stage="diarization", progress=55)
        logger.info("[INFO] Diarization started")
        diarization = diarize_audio(str(temp_wav))
        final_segments = align_transcript(transcription.segments, diarization.turns)
        logger.info("[INFO] Diarization & Alignment completed")

        if not diarization.available and diarization.message:
            warnings.append(diarization.message)

        speakers_list = sorted({seg.speaker for seg in final_segments if seg.speaker})
        detected_speakers_count = len(speakers_list) if diarization.available else 0
        is_single_speaker = detected_speakers_count <= 1
        transcript_text = format_transcript_text(final_segments, detected_speakers=detected_speakers_count)

        # 5. Quality Validation
        validation_report = validate_results(
            audio_duration=duration,
            detected_language=transcription.language,
            language_confidence=transcription.language_probability,
            segments=final_segments,
            diarization_available=diarization.available,
            speakers=speakers_list,
        )
        if validation_report.warnings:
            warnings.extend(validation_report.warnings)

        # 6. Summarization in selected language
        job_manager.update(job_id, stage="summarization", progress=70)
        logger.info("[INFO] Summarization started")
        summary, summary_method, summary_warning, summary_language, _gemini_err = summarize_transcript(
            transcription.text,
            output_language=job.output_language,
            source_language=transcription.language,
        )
        if summary_warning:
            warnings.append(summary_warning)

        language_matched = summary_language == job.output_language
        if not language_matched:
            logger.info(
                "[INFO] Summary language %s != requested %s; attempting translation",
                summary_language,
                job.output_language,
            )
            summary, translate_warning, translated_ok, _terr = translate_text(
                summary,
                source_language=summary_language,
                target_language=job.output_language,
            )
            if translate_warning:
                warnings.append(translate_warning)
            if translated_ok:
                summary_language = job.output_language
                language_matched = True
                if summary_method == "extractive":
                    summary_method = "extractive+gemini_translate"
            else:
                language_matched = False

        logger.info(
            "[INFO] Summary completed method=%s summary_language=%s matched=%s",
            summary_method,
            summary_language,
            language_matched,
        )

        # Persist text artifacts
        transcript_path = settings.transcripts_dir / f"{job_id}_transcript.txt"
        summary_path = settings.summaries_dir / f"{job_id}_summary.txt"
        write_text(transcript_path, transcript_text)
        write_text(summary_path, summary)

        # 7. Text-to-Speech from final summary
        job_manager.update(job_id, stage="tts", progress=88)
        logger.info("[INFO] TTS started")
        audio_summary_path = settings.audio_dir / f"{job_id}_summary.mp3"
        tts_language = summary_language
        tts_ok = False
        tts_method = None
        tts_error = None
        audio_summary_file = None

        if not language_matched:
            tts_error = (
                f"Audio summary was not generated in "
                f"{SUPPORTED_LANGUAGES.get(job.output_language, job.output_language)} "
                f"because the final summary language is "
                f"{SUPPORTED_LANGUAGES.get(summary_language, summary_language)}."
            )
            warnings.append(tts_error)
            logger.warning(tts_error)
        elif not language_supported_for_tts(tts_language):
            tts_error = (
                f"TTS does not support language '{tts_language}'. "
                "Text summary is still available."
            )
            warnings.append(tts_error)
        else:
            tts_ok, tts_method, tts_error = synthesize_speech(
                summary,
                tts_language,
                audio_summary_path,
            )
            if not tts_ok:
                warnings.append(tts_error or "Audio summary could not be generated.")
            else:
                audio_summary_file = str(audio_summary_path.name)

        finished = time.time()
        processing_time = round(finished - (job.started_at or finished), 2)

        # Deduplicate warnings
        unique_warnings = list(dict.fromkeys(warnings))

        result = {
            "job_id": job_id,
            "filename": job.original_filename,
            "duration_seconds": round(duration, 2),
            "duration_formatted": format_timestamp(duration),
            "detected_language": transcription.language,
            "detected_language_name": transcription.language_name,
            "detected_language_probability": round(transcription.language_probability, 3),
            "detected_confident": transcription.detected_confident,
            "detected_language_detail": transcription.language_detection.to_dict(),
            "detected_speakers": detected_speakers_count,
            "single_speaker": is_single_speaker,
            "speakers": speakers_list,
            "original_audio_url": f"/api/download/{job_id}/original",
            "output_language": job.output_language,
            "output_language_name": SUPPORTED_LANGUAGES.get(job.output_language, job.output_language),
            "processing_time_seconds": processing_time,
            "transcript": {
                "full_text": transcription.text,
                "diarized": diarization.available,
                "speaker_count": detected_speakers_count,
                "detected_speakers": detected_speakers_count,
                "single_speaker": is_single_speaker,
                "message": diarization.message,
                "segments": [s.to_dict() for s in final_segments],
                "formatted": transcript_text,
                "method": diarization.method,
                "overlap_supported": diarization.overlap_supported,
            },
            "validation": validation_report.to_dict(),
            "summary": {
                "text": summary,
                "method": summary_method,
                "language": summary_language,
                "summary_language": summary_language,
                "requested_language": job.output_language,
                "language_matched": language_matched,
            },
            "audio_summary": {
                "available": tts_ok,
                "method": tts_method,
                "filename": audio_summary_file,
                "url": f"/api/download/{job_id}/audio" if tts_ok else None,
                "message": None if tts_ok else (tts_error or "Audio summary unavailable."),
                "tts_language": tts_language if tts_ok else None,
                "requested_language": job.output_language,
            },
            "downloads": {
                "transcript": f"/api/download/{job_id}/transcript",
                "summary": f"/api/download/{job_id}/summary",
                "audio": f"/api/download/{job_id}/audio" if tts_ok else None,
            },
            "warnings": unique_warnings,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        job_manager.update(
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            finished_at=finished,
            warnings=unique_warnings,
            result=result,
            error=None,
            error_stage=None,
        )
        logger.info("[INFO] Job %s completed successfully", job_id)

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("[ERROR] Job %s failed: %s\n%s", job_id, exc, tb)
        message = str(exc) if str(exc) else "Processing failed due to an unexpected error."
        if "Traceback" in message:
            message = "Processing failed due to an unexpected error."
        current = job_manager.get(job_id)
        stage = current.stage if current else "processing"
        job_manager.update(
            job_id,
            status="failed",
            error=message,
            error_stage=stage,
            finished_at=time.time(),
            progress=STAGE_PROGRESS.get(stage, 0),
        )
    finally:
        cleanup_temp_files(temp_wav)
