"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
STORAGE_ROOT = BACKEND_ROOT / "storage"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    whisper_model: str = "medium"
    device: str = "cpu"
    max_audio_duration: int = 3600
    hf_token: str = ""
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def uploads_dir(self) -> Path:
        return STORAGE_ROOT / "uploads"

    @property
    def temp_dir(self) -> Path:
        return STORAGE_ROOT / "temp"

    @property
    def transcripts_dir(self) -> Path:
        return STORAGE_ROOT / "transcripts"

    @property
    def summaries_dir(self) -> Path:
        return STORAGE_ROOT / "summaries"

    @property
    def audio_dir(self) -> Path:
        return STORAGE_ROOT / "audio"

    @property
    def gemini_configured(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def ensure_storage_dirs() -> None:
    settings = get_settings()
    for path in (
        settings.uploads_dir,
        settings.temp_dir,
        settings.transcripts_dir,
        settings.summaries_dir,
        settings.audio_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "kn": "Kannada",
    "te": "Telugu",
}

LANGUAGE_NAME_TO_CODE = {v.lower(): k for k, v in SUPPORTED_LANGUAGES.items()}

ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mpeg", ".mp4", ".webm", ".ogg", ".flac"}
ALLOWED_CONTENT_TYPES = {
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "application/octet-stream",
}
