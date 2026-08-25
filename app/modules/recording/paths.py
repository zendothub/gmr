"""Filesystem layout helpers for continuous camera recordings.

Layout::

    {root}/                          # usually …/video_record
      counter_camera/
        12_08_26_2AM_5AM.mp4
      entry_camera/
        12_08_26_2AM_5AM.mp4

No database tables — paths only.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger


def camera_folder_name(camera_name: str) -> str:
    """Map DB camera name → folder under video_record.

    ``Apollo counter`` → ``counter_camera``
    ``Apollo entry``   → ``entry_camera``
    """
    n = (camera_name or "").strip().lower()
    if "counter" in n:
        return "counter_camera"
    if "entry" in n:
        return "entry_camera"
    slug = re.sub(r"[^a-z0-9]+", "_", n).strip("_") or "unknown"
    if not slug.endswith("_camera"):
        slug = f"{slug}_camera"
    return slug


def _fmt_hour_ampm(dt: datetime) -> str:
    """Format hour as ``2AM`` / ``12PM`` (no leading zero)."""
    h24 = dt.hour
    suffix = "AM" if h24 < 12 else "PM"
    h12 = h24 % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}{suffix}"


def chunk_window(now: datetime, chunk_hours: float) -> Tuple[datetime, datetime]:
    """Wall-clock chunk containing ``now``, aligned from local midnight."""
    if chunk_hours <= 0:
        chunk_hours = 3.0
    local = now.astimezone() if now.tzinfo else now
    day_start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    slot_secs = float(chunk_hours) * 3600.0
    elapsed = (local - day_start).total_seconds()
    slot = int(elapsed // slot_secs)
    start = day_start + timedelta(seconds=slot * slot_secs)
    end = start + timedelta(seconds=slot_secs)
    return start, end


def recording_filename(start: datetime, end: datetime) -> str:
    """``dd_mm_yy_starttime_endtime.mp4`` e.g. ``12_08_26_2AM_5AM.mp4``."""
    date_part = start.strftime("%d_%m_%y")
    return f"{date_part}_{_fmt_hour_ampm(start)}_{_fmt_hour_ampm(end)}.mp4"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_recording_root(
    explicit_root: str,
    hdd_mount: str,
    fallback_root: str,
    root_folder_name: str = "video_record",
) -> Path:
    """Pick writable recording root without touching existing app paths.

    Priority:
      1. ``RECORDING_ROOT`` if set and writable (created if missing)
      2. ``{RECORDING_HDD_MOUNT}/{root_folder_name}`` if mount is live
      3. Expanded ``RECORDING_FALLBACK_ROOT`` (Desktop by default)
    """
    def _as_video_record(base: Path) -> Path:
        if base.name == root_folder_name:
            return base
        return base / root_folder_name

    def _try(path: Path, label: str) -> Optional[Path]:
        try:
            ensure_dir(path)
            probe = path / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            logger.info(f"Recording root ({label}): {path}")
            return path
        except Exception as e:
            logger.warning(f"Recording root not usable ({label}={path}): {e}")
            return None

    if explicit_root and explicit_root.strip():
        p = _as_video_record(Path(os.path.expanduser(explicit_root.strip())).resolve())
        got = _try(p, "RECORDING_ROOT")
        if got is not None:
            return got

    mount = Path(os.path.expanduser(hdd_mount or "/mnt/video_hdd"))
    if mount.is_dir() and os.path.ismount(str(mount)):
        got = _try(_as_video_record(mount), "HDD mount")
        if got is not None:
            return got
    else:
        logger.info(
            f"Recording HDD not mounted at {mount} — "
            f"run scripts/mount_recording_hdd.sh or set RECORDING_ROOT"
        )

    fb = Path(os.path.expanduser(fallback_root or "~/Desktop/video_record")).resolve()
    got = _try(_as_video_record(fb if fb.name == root_folder_name else fb), "Desktop fallback")
    if got is not None:
        return got

    # Last resort: cwd/video_record (should rarely hit)
    last = Path.cwd() / root_folder_name
    ensure_dir(last)
    logger.warning(f"Recording root fallback to CWD: {last}")
    return last


def camera_output_path(
    root: Path,
    camera_name: str,
    start: datetime,
    end: datetime,
) -> Path:
    folder = ensure_dir(root / camera_folder_name(camera_name))
    return folder / recording_filename(start, end)
