"""Upload API routes."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from app.config import ALLOWED_EXTENSIONS, get_settings
from app.utils.audio import AudioError, check_ffmpeg, format_timestamp, get_audio_duration
from app.utils.files import generate_safe_filename, validate_extension

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

# Simple in-memory upload registry
UPLOADS: dict[str, dict] = {}


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    settings = get_settings()

    ok, ffmpeg_msg = check_ffmpeg()
    if not ok:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": ffmpeg_msg,
                "stage": "ffmpeg",
            },
        )

    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No filename provided. Please choose an audio file.",
                "stage": "upload",
            },
        )

    if not validate_extension(file.filename):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": (
                    f"Unsupported file type. Allowed extensions: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                ),
                "stage": "upload",
            },
        )

    upload_id = uuid.uuid4().hex
    safe_name = generate_safe_filename(file.filename)
    dest = settings.uploads_dir / f"{upload_id}_{safe_name}"

    try:
        data = await file.read()
        if not data:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Uploaded file is empty.",
                    "stage": "upload",
                },
            )
        # Soft size guard ~200MB
        if len(data) > 200 * 1024 * 1024:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "File is too large. Please upload an audio file under 200 MB.",
                    "stage": "upload",
                },
            )
        dest.write_bytes(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to save upload")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Could not save uploaded file: {exc}",
                "stage": "upload",
            },
        )

    duration = None
    duration_formatted = None
    try:
        duration = get_audio_duration(dest)
        duration_formatted = format_timestamp(duration)
        if duration > settings.max_audio_duration:
            dest.unlink(missing_ok=True)
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": (
                        f"Audio duration ({duration / 60:.1f} min) exceeds the maximum "
                        f"of {settings.max_audio_duration // 60} minutes."
                    ),
                    "stage": "upload",
                },
            )
    except AudioError as exc:
        # Invalid audio — reject
        dest.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": exc.message,
                "stage": exc.stage,
            },
        )

    meta = {
        "upload_id": upload_id,
        "original_filename": Path(file.filename).name,
        "stored_filename": dest.name,
        "stored_path": str(dest),
        "size_bytes": dest.stat().st_size,
        "duration_seconds": duration,
        "duration_formatted": duration_formatted,
    }
    UPLOADS[upload_id] = meta
    logger.info("[INFO] Upload saved: %s (%s)", upload_id, meta["original_filename"])

    return {
        "success": True,
        "upload_id": upload_id,
        "filename": meta["original_filename"],
        "size_bytes": meta["size_bytes"],
        "duration_seconds": duration,
        "duration_formatted": duration_formatted,
    }
