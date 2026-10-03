"""Processing, status, result, and download routes."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import SUPPORTED_LANGUAGES, get_settings
from app.routes.upload import UPLOADS
from app.services.jobs import job_manager, start_processing

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["processing"])


class ProcessRequest(BaseModel):
    upload_id: str = Field(..., description="ID returned by /api/upload")
    output_language: str = Field("en", description="en | hi | kn | te")


@router.post("/process")
async def process_podcast(body: ProcessRequest):
    output_language = (body.output_language or "en").lower().strip()
    if output_language not in SUPPORTED_LANGUAGES:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Invalid output language. Choose English, Hindi, Kannada, or Telugu.",
                "stage": "process",
            },
        )

    upload = UPLOADS.get(body.upload_id)
    if not upload:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Upload not found. Please upload the audio file again.",
                "stage": "process",
            },
        )

    stored = Path(upload["stored_path"])
    if not stored.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Uploaded file is missing on the server. Please upload again.",
                "stage": "process",
            },
        )

    job = job_manager.create_job(
        upload_id=body.upload_id,
        stored_path=str(stored),
        original_filename=upload["original_filename"],
        output_language=output_language,
    )
    start_processing(job.job_id)
    logger.info("[INFO] Job %s queued for processing", job.job_id)

    return {
        "success": True,
        "job_id": job.job_id,
        "status": "processing",
    }


@router.get("/status/{job_id}")
async def get_status(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Job not found.",
                "stage": "status",
            },
        )
    payload = job.to_status()
    payload["success"] = True
    return payload


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    job = job_manager.get(job_id)
    if not job:
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Job not found.",
                "stage": "result",
            },
        )
    if job.status == "failed":
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": job.error or "Processing failed.",
                "stage": job.error_stage or job.stage,
                "job_id": job_id,
            },
        )
    if job.status != "completed" or not job.result:
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "error": "Processing is not complete yet.",
                "stage": job.stage,
                "status": job.status,
                "progress": job.progress,
            },
        )
    return {"success": True, **job.result}


@router.get("/download/{job_id}/transcript")
async def download_transcript(job_id: str):
    settings = get_settings()
    path = settings.transcripts_dir / f"{job_id}_transcript.txt"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Transcript not found.", "stage": "download"},
        )
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=f"{job_id}_transcript.txt",
    )


@router.get("/download/{job_id}/summary")
async def download_summary(job_id: str):
    settings = get_settings()
    path = settings.summaries_dir / f"{job_id}_summary.txt"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "Summary not found.", "stage": "download"},
        )
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=f"{job_id}_summary.txt",
    )


@router.get("/download/{job_id}/audio")
async def download_audio(job_id: str):
    settings = get_settings()
    path = settings.audio_dir / f"{job_id}_summary.mp3"
    if not path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "error": "Audio summary not found.",
                "stage": "download",
            },
        )
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=f"{job_id}_summary.mp3",
    )


@router.get("/download/{job_id}/original")
async def download_original_audio(job_id: str):
    """Serve the submitted audio for timestamp-synchronised transcript playback."""
    job = job_manager.get(job_id)
    path = Path(job.stored_path) if job and job.stored_path else None
    if not path or not path.exists():
        return JSONResponse(status_code=404, content={"success": False, "error": "Original audio not found."})
    return FileResponse(path, filename=job.original_filename or path.name)
