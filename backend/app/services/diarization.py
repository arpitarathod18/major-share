"""Voice-embedding speaker diarization with PyAnnote Community-1 pipeline."""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================================
# Runtime Shims for Torchaudio 2.11+, PyTorch 2.6+, SpeechBrain, HuggingFace Hub
# ============================================================================
_SHIMS_APPLIED = False


def _apply_runtime_shims() -> None:
    global _SHIMS_APPLIED
    if _SHIMS_APPLIED:
        return

    # 1. Torchaudio soundfile backend & AudioMetaData shim (needed for pyannote.audio.core.io)
    try:
        import soundfile as sf
        import torch
        import torchaudio

        @dataclasses.dataclass
        class AudioMetaData:
            sample_rate: int
            num_frames: int
            num_channels: int
            bits_per_sample: int = 16
            encoding: str = "PCM_S"

        if not hasattr(torchaudio, "AudioMetaData"):
            torchaudio.AudioMetaData = AudioMetaData

        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        orig_torchaudio_load = getattr(torchaudio, "load", None)

        def _safe_torchaudio_load(filepath, frame_offset=0, num_frames=-1, *args, **kwargs):
            if isinstance(filepath, dict):
                filepath = filepath.get("audio", filepath)
            try:
                start = max(0, frame_offset)
                stop = start + num_frames if num_frames > 0 else None
                data, samplerate = sf.read(str(filepath), start=start, stop=stop, dtype="float32")
                if data.ndim == 1:
                    tensor = torch.from_numpy(data).unsqueeze(0)
                else:
                    tensor = torch.from_numpy(data.T)
                return tensor, samplerate
            except Exception:
                if orig_torchaudio_load:
                    return orig_torchaudio_load(filepath, frame_offset=frame_offset, num_frames=num_frames, *args, **kwargs)
                raise

        def _safe_torchaudio_info(filepath, *args, **kwargs):
            if isinstance(filepath, dict):
                filepath = filepath.get("audio", filepath)
            info = sf.info(str(filepath))
            return AudioMetaData(
                sample_rate=info.samplerate,
                num_frames=info.frames,
                num_channels=info.channels,
                bits_per_sample=16,
                encoding="PCM_S",
            )

        torchaudio.load = _safe_torchaudio_load
        torchaudio.info = _safe_torchaudio_info
    except Exception as exc:
        logger.debug("[DIARIZATION] Torchaudio shim notice: %s", exc)

    # 2. PyTorch 2.6+ checkpoint loading compatibility (weights_only=False for PyAnnote)
    try:
        import torch

        orig_torch_load = torch.load

        def _compat_torch_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return orig_torch_load(*args, **kwargs)

        torch.load = _compat_torch_load
    except Exception as exc:
        logger.debug("[DIARIZATION] PyTorch load shim notice: %s", exc)

    # 3. SpeechBrain LazyModule safety for inspect.stack()
    try:
        import speechbrain.utils.importutils as sb_import

        orig_getattr = sb_import.LazyModule.__getattr__

        def _safe_getattr(self, attr):
            if attr == "__file__":
                return None
            return orig_getattr(self, attr)

        sb_import.LazyModule.__getattr__ = _safe_getattr
    except Exception:
        pass

    # 4. HuggingFace Hub token argument translation (use_auth_token -> token)
    try:
        import huggingface_hub

        orig_download = huggingface_hub.hf_hub_download

        def _patched_download(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return orig_download(*args, **kwargs)

        huggingface_hub.hf_hub_download = _patched_download

        orig_snapshot = huggingface_hub.snapshot_download

        def _patched_snapshot(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs["token"] = kwargs.pop("use_auth_token")
            return orig_snapshot(*args, **kwargs)

        huggingface_hub.snapshot_download = _patched_snapshot
    except Exception as exc:
        logger.debug("[DIARIZATION] HuggingFace Hub shim notice: %s", exc)

    _SHIMS_APPLIED = True


# Apply shims immediately at module import
_apply_runtime_shims()

# Pipeline singleton & cached status
_pipeline: Any | None = None
_pipeline_method: str = "none"
_pipeline_error: str | None = None
COMMUNITY_MODEL_ID = "pyannote/speaker-diarization-community-1"


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class DiarizationResult:
    turns: list[SpeakerTurn]
    speaker_count: int
    available: bool
    message: str | None = None
    method: str = "none"
    overlap_supported: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": [turn.__dict__ for turn in self.turns],
            "speaker_count": self.speaker_count,
            "available": self.available,
            "message": self.message,
            "method": self.method,
            "overlap_supported": self.overlap_supported,
        }


def _find_cached_snapshot(model_repo: str) -> Path | None:
    """Check for downloaded Hugging Face snapshot models in ~/.cache."""
    cache_dir = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    folder_name = f"models--{model_repo.replace('/', '--')}"
    target = cache_dir / folder_name / "snapshots"
    if target.exists():
        for snap in target.glob("*"):
            if snap.is_dir():
                model_bin = snap / "pytorch_model.bin"
                if model_bin.exists():
                    return model_bin
                config_yaml = snap / "config.yaml"
                if config_yaml.exists():
                    return config_yaml
    return None


def _get_pyannote_pipeline() -> Any:
    """Load and cache the PyAnnote Community-1 speaker diarization pipeline."""
    global _pipeline, _pipeline_method, _pipeline_error
    if _pipeline is not None:
        return _pipeline

    _apply_runtime_shims()

    from app.config import get_settings

    settings = get_settings()
    token = settings.hf_token.strip() or os.environ.get("HF_TOKEN", "").strip() or None

    # Strategy 1: Load official Community-1 pipeline (recommended)
    try:
        from pyannote.audio import Pipeline

        logger.info("[DIARIZATION] Loading PyAnnote Community-1 pipeline (%s)...", COMMUNITY_MODEL_ID)
        pipe = Pipeline.from_pretrained(COMMUNITY_MODEL_ID, token=token)
        if pipe is not None:
            _pipeline = pipe
            _pipeline_method = COMMUNITY_MODEL_ID
            _pipeline_error = None
            logger.info("[DIARIZATION] %s initialized successfully.", COMMUNITY_MODEL_ID)
            return _pipeline
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.warning("[DIARIZATION] Loading %s failed: %s", COMMUNITY_MODEL_ID, exc)

    # Strategy 2: Fallback to cached local weights if available (e.g. segmentation-3.0 + wespeaker)
    seg_cached = _find_cached_snapshot("pyannote/segmentation-3.0")
    emb_cached = _find_cached_snapshot("pyannote/wespeaker-voxceleb-resnet34-LM")

    if seg_cached and emb_cached:
        try:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import SpeakerDiarization

            logger.info("[DIARIZATION] Attempting fallback to local cached segmentation & embedding models...")
            seg_model = Model.from_pretrained(seg_cached)
            emb_model = Model.from_pretrained(emb_cached)
            pipe = SpeakerDiarization(
                segmentation=seg_model,
                embedding=emb_model,
                clustering="AgglomerativeClustering",
                embedding_batch_size=32,
                embedding_exclude_overlap=True,
                segmentation_batch_size=32,
            )
            pipe.instantiate(
                {
                    "clustering": {
                        "method": "centroid",
                        "min_cluster_size": 12,
                        "threshold": 0.7045654963945799,
                    },
                    "segmentation": {"min_duration_off": 0.0},
                }
            )
            _pipeline = pipe
            _pipeline_method = "pyannote_local_cached"
            _pipeline_error = None
            logger.info("[DIARIZATION] PyAnnote fallback pipeline initialized from local cache.")
            return _pipeline
        except Exception as exc:
            logger.warning("[DIARIZATION] Local cached fallback load failed: %s", exc)

    err_msg = (
        f"PyAnnote speaker diarization model could not be initialized ({_pipeline_error or 'missing credentials'}). "
        f"Please accept terms at https://hf.co/{COMMUNITY_MODEL_ID} and set HF_TOKEN in backend/.env."
    )
    raise RuntimeError(err_msg)


def diarization_status() -> dict[str, Any]:
    """Return diarization availability and detailed status dictionary."""
    _apply_runtime_shims()
    from app.config import get_settings

    settings = get_settings()
    token = settings.hf_token.strip() or os.environ.get("HF_TOKEN", "").strip()

    if _pipeline is not None:
        return {
            "available": True,
            "provider": "pyannote",
            "model": _pipeline_method,
            "reason": None,
        }

    # If token is provided, probe/check availability
    if token:
        try:
            pipe = _get_pyannote_pipeline()
            if pipe is not None:
                return {
                    "available": True,
                    "provider": "pyannote",
                    "model": _pipeline_method,
                    "reason": None,
                }
        except Exception as exc:
            return {
                "available": False,
                "provider": "pyannote",
                "model": COMMUNITY_MODEL_ID,
                "reason": str(exc),
            }

    # Check local cached snapshot availability
    seg_cached = _find_cached_snapshot("pyannote/segmentation-3.0")
    emb_cached = _find_cached_snapshot("pyannote/wespeaker-voxceleb-resnet34-LM")
    if seg_cached and emb_cached:
        return {
            "available": True,
            "provider": "pyannote",
            "model": "pyannote_local_cached",
            "reason": None,
        }

    return {
        "available": False,
        "provider": "pyannote",
        "model": COMMUNITY_MODEL_ID,
        "reason": f"HF_TOKEN is not configured in backend/.env. Visit https://hf.co/{COMMUNITY_MODEL_ID} to accept terms and get a token.",
    }


def _stable_turns(raw_turns: list[tuple[float, float, str]]) -> list[SpeakerTurn]:
    """Map raw model speaker labels to stable, chronological IDs (SPEAKER_00, SPEAKER_01).

    Also applies temporal smoothing:
      - Filters out micro-fragments (< 0.20s).
      - Merges adjacent continuous turns for the same speaker (gap <= 0.25s).
    """
    raw_turns.sort(key=lambda item: (item[0], item[1], item[2]))
    label_map: dict[str, str] = {}
    turns: list[SpeakerTurn] = []

    for start, end, raw_label in raw_turns:
        if end - start < 0.20:
            continue

        label_map.setdefault(raw_label, f"SPEAKER_{len(label_map):02d}")
        speaker = label_map[raw_label]

        if turns and turns[-1].speaker == speaker and start - turns[-1].end <= 0.25:
            turns[-1] = SpeakerTurn(
                start=turns[-1].start,
                end=max(turns[-1].end, round(end, 3)),
                speaker=speaker,
            )
        else:
            turns.append(SpeakerTurn(start=round(start, 3), end=round(end, 3), speaker=speaker))

    return turns


def diarize_audio(audio_path: str) -> DiarizationResult:
    """Generate speaker turns based on voice characteristics, independent of language.

    Returns:
      DiarizationResult containing stable SpeakerTurn list, speaker count, and method.
      If diarization fails, returns turns=[] and available=False without fake labels.
    """
    try:
        pipeline = _get_pyannote_pipeline()
        logger.info("[DIARIZATION] Running voice diarization on %s...", Path(audio_path).name)
        annotation = pipeline(audio_path)

        raw_turns: list[tuple[float, float, str]] = []

        # PyAnnote Community-1 provides exclusive_speaker_diarization for STT alignment
        if hasattr(annotation, "exclusive_speaker_diarization") and annotation.exclusive_speaker_diarization is not None:
            exclusive_output = annotation.exclusive_speaker_diarization
            if hasattr(exclusive_output, "itertracks"):
                for turn, _, label in exclusive_output.itertracks(yield_label=True):
                    if float(turn.end) > float(turn.start):
                        raw_turns.append((float(turn.start), float(turn.end), str(label)))
            else:
                for item in exclusive_output:
                    if len(item) == 2:
                        turn, label = item
                        raw_turns.append((float(turn.start), float(turn.end), str(label)))
                    elif len(item) == 3:
                        turn, _, label = item
                        raw_turns.append((float(turn.start), float(turn.end), str(label)))
        elif hasattr(annotation, "speaker_diarization") and annotation.speaker_diarization is not None:
            spk_output = annotation.speaker_diarization
            if hasattr(spk_output, "itertracks"):
                for turn, _, label in spk_output.itertracks(yield_label=True):
                    if float(turn.end) > float(turn.start):
                        raw_turns.append((float(turn.start), float(turn.end), str(label)))
        elif hasattr(annotation, "itertracks"):
            for turn, _, label in annotation.itertracks(yield_label=True):
                if float(turn.end) > float(turn.start):
                    raw_turns.append((float(turn.start), float(turn.end), str(label)))

        turns = _stable_turns(raw_turns)
        if not turns:
            raise RuntimeError("The diarization model found no valid speech turns.")

        speakers = sorted({turn.speaker for turn in turns})
        speaker_durations = {spk: 0.0 for spk in speakers}
        for t in turns:
            speaker_durations[t.speaker] += max(0.0, t.end - t.start)

        log_details = "\n".join(
            f"  {spk}: {dur:.2f}s" for spk, dur in sorted(speaker_durations.items())
        )
        logger.info(
            "[DIARIZATION]\n"
            "  Detected speakers: %d\n"
            "  Turns: %d\n"
            "  Method: %s\n"
            "%s",
            len(speakers),
            len(turns),
            _pipeline_method,
            log_details,
        )

        return DiarizationResult(
            turns=turns,
            speaker_count=len(speakers),
            available=True,
            method=_pipeline_method,
            overlap_supported=True,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("[DIARIZATION] Diarization unavailable: %s", exc)
        return DiarizationResult(
            turns=[],
            speaker_count=0,
            available=False,
            message=(
                "Voice-embedding diarization was unavailable; speaker labels were not assigned. "
                "Ensure PyAnnote models are installed or set HF_TOKEN in backend/.env."
            ),
            method="none",
            overlap_supported=False,
        )
