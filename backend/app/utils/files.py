"""File utilities: safe names, validation, cleanup."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from app.config import ALLOWED_EXTENSIONS

logger = logging.getLogger(__name__)


def generate_safe_filename(original_name: str) -> str:
    """Return a unique safe filename preserving a validated extension."""
    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".wav"
    return f"{uuid.uuid4().hex}{ext}"


def validate_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_temp_files(*paths: Path | str | None) -> None:
    """Remove temporary files/directories that are no longer needed."""
    for item in paths:
        if not item:
            continue
        path = Path(item)
        try:
            if path.is_file():
                path.unlink(missing_ok=True)
            elif path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
        except OSError as exc:
            logger.warning("Could not clean up %s: %s", path, exc)


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
