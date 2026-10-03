"""FastAPI application entrypoint for Meri Awaaz Hi Pehchan Hai."""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import ensure_storage_dirs, get_settings
from app.routes import processing, upload
from app.services.diarization import diarization_status
from app.services.gemini_client import get_gemini_model_name, probe_gemini
from app.services.transcription import whisper_status
from app.services.tts import tts_status
from app.utils.audio import check_ffmpeg

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("meri_awaaz")


def _startup_validation() -> dict:
    settings = get_settings()
    ensure_storage_dirs()
    # Clear settings cache so GEMINI_MODEL changes are picked up on reload
    get_settings.cache_clear()
    settings = get_settings()

    ffmpeg_ok, _ = check_ffmpeg()
    tts = tts_status()
    info = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "ffmpeg": ffmpeg_ok,
        "whisper": whisper_status(),
        "diarization": diarization_status(),
        "gemini_configured": settings.gemini_configured,
        "gemini_model": get_gemini_model_name(),
        "tts": tts.get("available"),
        "device": settings.device,
        "whisper_model": settings.whisper_model,
    }

    logger.info("Startup validation: %s", info)
    if not info["ffmpeg"]:
        logger.warning("FFmpeg is not available. Audio processing will fail until installed.")
    if not settings.gemini_configured:
        logger.warning("GEMINI_API_KEY not set. Multilingual Gemini summaries will be unavailable.")
    else:
        logger.info("[Gemini] API key configured: true")
        logger.info("[Gemini] Model: %s", get_gemini_model_name())
    if info["whisper"] != "available":
        logger.warning("faster-whisper is not importable yet.")
    return info


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _startup_validation()
    yield


app = FastAPI(
    title="Meri Awaaz Hi Pehchan Hai",
    description=(
        "Multilingual Podcast Processing System for India — "
        "transcription, speaker diarization, summarization, and TTS"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(processing.router)


@app.get("/api/health")
async def health():
    settings = get_settings()
    ffmpeg_ok, ffmpeg_detail = check_ffmpeg()
    whisper = whisper_status()
    diar = diarization_status()
    tts = tts_status()

    gemini_info = {
        "configured": settings.gemini_configured,
        "available": False,
        "model": get_gemini_model_name(),
    }
    # Lightweight availability: configured + model name present.
    # Full live probe is on /api/health/gemini to avoid slowing every health poll.
    if settings.gemini_configured:
        gemini_info["available"] = True

    return {
        "backend": "ok",
        "whisper": {"available": whisper == "available", "model": settings.whisper_model},
        "ffmpeg": {
            "available": ffmpeg_ok,
            "detail": ffmpeg_detail if not ffmpeg_ok else "available",
        },
        "gemini": gemini_info,
        "diarization": diar,
        "tts": tts,
        "device": settings.device,
        "supported_languages": ["en", "hi", "kn", "te"],
    }


@app.get("/api/health/gemini")
async def health_gemini():
    """Live Gemini probe (small request). Does not expose API keys."""
    return {"success": True, **probe_gemini("kn")}


@app.get("/")
async def root():
    return {
        "name": "Meri Awaaz Hi Pehchan Hai",
        "message": "Multilingual Podcast Processing System for India",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception):
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "An unexpected server error occurred. Please try again.",
            "stage": "server",
        },
    )
