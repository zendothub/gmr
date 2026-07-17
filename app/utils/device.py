"""Device detection utility.

Probes hardware once at application startup (CUDA → MPS → CPU) and stores the
result both in memory (module-level singleton) and on disk
(``runtime/device_info.json``).

Every AI component that needs a torch device string should call
``get_device()`` instead of running their own ``torch.cuda.is_available()``
checks — this guarantees a single consistent device decision across the entire
process.

Typical usage
-------------
In ``lifecycle.py`` startup::

    from app.utils.device import detect_and_save_device
    detect_and_save_device()

In model loaders (yolo_detector, osnet_extractor, insightface_analyzer)::

    from app.utils.device import get_device
    device = get_device()   # "cuda" | "mps" | "cpu"
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_DEVICE: Optional[str] = None
_DEVICE_INFO: dict = {}
_NVENC_AVAILABLE: bool = False   # set to True only if ffmpeg reports h264_nvenc

# Path for the on-disk runtime file
_RUNTIME_DIR = "runtime"
_DEVICE_FILE = os.path.join(_RUNTIME_DIR, "device_info.json")


def detect_and_save_device() -> str:
    """Probe hardware (CUDA → MPS → CPU), cache the result, and write to disk.

    This function is idempotent — if called more than once it returns the
    cached value without re-probing.

    Returns:
        The detected device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    global _DEVICE, _DEVICE_INFO

    if _DEVICE is not None:
        return _DEVICE

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        mps_available = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

        if cuda_available:
            _DEVICE = "cuda"
            cuda_device_name = torch.cuda.get_device_name(0)
            cuda_device_count = torch.cuda.device_count()
        elif mps_available:
            _DEVICE = "mps"
            cuda_device_name = None
            cuda_device_count = 0
        else:
            _DEVICE = "cpu"
            cuda_device_name = None
            cuda_device_count = 0

        _DEVICE_INFO = {
            "device": _DEVICE,
            "cuda_available": cuda_available,
            "mps_available": mps_available,
            "cuda_device_name": cuda_device_name,
            "cuda_device_count": cuda_device_count,
            "torch_version": torch.__version__,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

    except ImportError:
        logger.warning("torch not installed — defaulting device to 'cpu'")
        _DEVICE = "cpu"
        _DEVICE_INFO = {
            "device": "cpu",
            "cuda_available": False,
            "mps_available": False,
            "cuda_device_name": None,
            "cuda_device_count": 0,
            "torch_version": None,
            "detected_at": datetime.now(timezone.utc).isoformat(),
        }

    # Log the result clearly
    _log_device_info()

    # Persist to disk so external tools / monitoring can inspect it
    _write_device_file()

    return _DEVICE


def get_device() -> str:
    """Return the detected device string.

    If ``detect_and_save_device()`` has not been called yet, runs it now.
    This makes the function safe to call from any model loader even if the
    lifecycle startup hook was missed.

    Returns:
        ``"cuda"``, ``"mps"``, or ``"cpu"``
    """
    if _DEVICE is None:
        return detect_and_save_device()
    return _DEVICE


def get_device_info() -> dict:
    """Return the full device info dict (read-only copy)."""
    if not _DEVICE_INFO:
        detect_and_save_device()
    return dict(_DEVICE_INFO)


def insightface_ctx_id() -> int:
    """Return the InsightFace context id.

    InsightFace maps ctx_id to ONNX Runtime providers internally:
      ctx_id >= 0  →  tries CUDAExecutionProvider first
      ctx_id < 0   →  CPUExecutionProvider only

    For MPS we pass ctx_id=0 so the session is constructed, then we override
    the providers list separately via ``get_insightface_providers()``.
    """
    device = get_device()
    # Use ctx_id=0 for both CUDA and MPS (CPU uses -1)
    return 0 if device in ("cuda", "mps") else -1


def get_insightface_providers() -> list:
    """Return the ONNX Runtime execution providers list for InsightFace.

    Priority:
      CUDA   → CUDAExecutionProvider   (NVIDIA GPU)
      MPS    → CoreMLExecutionProvider (Apple Neural Engine / GPU)
      CPU    → CPUExecutionProvider    (software fallback)

    Always include CPUExecutionProvider as final fallback.
    """
    device = get_device()
    if device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "mps":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def get_ffmpeg_video_codec_args(ffmpeg_binary: str = "ffmpeg") -> list:
    """Return the FFmpeg video codec CLI argument list for the current device.

    Priority:
      1. CUDA  → h264_nvenc (probed — falls back to libx264 if not in ffmpeg build)
      2. MPS   → h264_videotoolbox  (Apple VideoToolbox hardware encoder)
      3. CPU   → libx264  (software, veryfast + zerolatency)

    All paths include explicit bitrate control (STREAM_BITRATE) because FFmpeg
    defaults to CRF-based quality encoding without -b:v, producing 5-8 Mbps for
    1080p surveillance — 4-6× heavier than the ~1.2 Mbps this pipeline targets.

    The returned list is ready to splice into an ffmpeg command, e.g.::

        cmd = [..., "-i", source] + get_ffmpeg_video_codec_args() + ["-f", "rtsp", url]
    """
    from app.config import get_settings
    settings = get_settings()
    bitrate = settings.STREAM_BITRATE
    max_height = settings.STREAM_MAX_HEIGHT

    device = get_device()

    # Build scale filter if max_height > 0 (0 = passthrough native resolution)
    scale_filter = [f"scale=-2:{max_height}"] if max_height > 0 else []

    if device == "cuda":
        if _NVENC_AVAILABLE or _probe_nvenc(ffmpeg_binary):
            logger.debug("FFmpeg encoder: h264_nvenc (CUDA)")
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p4",       # NVENC balanced preset (good quality / speed)
                "-tune", "ll",         # low-latency tune
                *scale_filter,
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", str(int(bitrate.rstrip("k")) * 2) + "k",
                "-pix_fmt", "yuv420p",
            ]
        # CUDA GPU found but ffmpeg was built without NVENC — fall through to libx264
        logger.warning("CUDA GPU detected but h264_nvenc not available in ffmpeg — falling back to libx264")

    if device == "mps":
        logger.debug("FFmpeg encoder: h264_videotoolbox (MPS/Apple Silicon)")
        return [
            "-c:v", "h264_videotoolbox",
            *scale_filter,
            "-b:v", bitrate,
            "-maxrate", bitrate,
            "-pix_fmt", "yuv420p",
        ]

    # CPU (or CUDA without NVENC)
    logger.debug("FFmpeg encoder: libx264 (CPU software)")
    return [
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        *scale_filter,
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", str(int(bitrate.rstrip("k")) * 2) + "k",
        "-pix_fmt", "yuv420p",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _probe_nvenc(ffmpeg_binary: str = "ffmpeg") -> bool:
    """Return True if h264_nvenc encoder is present in the installed ffmpeg."""
    global _NVENC_AVAILABLE
    try:
        import subprocess
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        available = "h264_nvenc" in result.stdout
        _NVENC_AVAILABLE = available
        return available
    except Exception:
        _NVENC_AVAILABLE = False
        return False

def _log_device_info() -> None:
    info = _DEVICE_INFO
    device = info.get("device", "unknown")
    torch_ver = info.get("torch_version", "?")

    if device == "cuda":
        name = info.get("cuda_device_name", "unknown GPU")
        count = info.get("cuda_device_count", 1)
        logger.info(
            f"🚀 Device detected: CUDA  "
            f"[{name}] × {count} GPU(s)  (torch {torch_ver})"
        )
    elif device == "mps":
        logger.info(
            f"🍎 Device detected: MPS (Apple Silicon)  (torch {torch_ver})"
        )
    else:
        logger.info(
            f"💻 Device detected: CPU  (torch {torch_ver})"
        )


def _write_device_file() -> None:
    """Write _DEVICE_INFO to runtime/device_info.json."""
    try:
        os.makedirs(_RUNTIME_DIR, exist_ok=True)
        with open(_DEVICE_FILE, "w", encoding="utf-8") as f:
            json.dump(_DEVICE_INFO, f, indent=2)
        logger.debug(f"Device info written to {_DEVICE_FILE}")
    except Exception as exc:
        logger.warning(f"Could not write device_info.json: {exc}")
