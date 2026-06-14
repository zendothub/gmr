"""MinIO client singleton — wraps the `minio` Python SDK.

Provides upload/download/delete operations for the MinIO bucket used to
store snapshots, crops, clips, and reports.  All binary data flows through
this module; the rest of the codebase never touches the local filesystem
for storage.
"""

from __future__ import annotations

import io
from typing import Optional

import numpy as np
from loguru import logger
from minio import Minio
from minio.error import S3Error

from app.config import get_settings

# Module-level singleton — initialised once at import time.
_settings = get_settings()
_client: Optional[Minio] = (
    Minio(
        _settings.MINIO_ENDPOINT,
        access_key=_settings.MINIO_ACCESS_KEY,
        secret_key=_settings.MINIO_SECRET_KEY,
        secure=_settings.MINIO_SECURE,
    )
    if _settings.MINIO_ENDPOINT
    else None
)

BUCKET_PREFIX = _settings.MINIO_BUCKET_PREFIX


def get_client() -> Minio:
    """Return the shared MinIO client, or raise if not configured."""
    if _client is None:
        raise RuntimeError("MinIO is not configured (MINIO_ENDPOINT is empty)")
    _ensure_bucket()
    return _client


def _ensure_bucket() -> None:
    """Create the bucket if it doesn't exist."""
    try:
        if not _client.bucket_exists(BUCKET_PREFIX):
            _client.make_bucket(BUCKET_PREFIX)
            logger.info(f"MinIO bucket '{BUCKET_PREFIX}' created")
    except S3Error as e:
        logger.error(f"MinIO bucket check failed: {e}")


# ---------------------------------------------------------------------------
# Public helpers used by StorageService
# ---------------------------------------------------------------------------

def upload_image(
    image: np.ndarray,
    object_name: str,
    quality: int = 85,
    content_type: str = "image/jpeg",
) -> Optional[str]:
    """Encode a numpy image to JPEG bytes and upload to MinIO.

    Returns the full object path (bucket/object_name) on success, or None.
    """
    try:
        import cv2
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return _put_object(object_name, buf.tobytes(), content_type, len(buf))
    except Exception as e:
        logger.error(f"MinIO upload_image failed: {e}")
        return None


def upload_bytes(
    data: bytes,
    object_name: str,
    content_type: str = "application/octet-stream",
) -> Optional[str]:
    """Upload raw bytes to MinIO."""
    return _put_object(object_name, data, content_type, len(data))


def download_bytes(object_name: str) -> Optional[bytes]:
    """Download an object from MinIO as bytes."""
    try:
        client = get_client()
        response = client.get_object(BUCKET_PREFIX, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except S3Error as e:
        logger.error(f"MinIO download failed for {object_name}: {e}")
        return None


def delete_object(object_name: str) -> bool:
    """Delete an object from MinIO. Returns True on success."""
    try:
        get_client().remove_object(BUCKET_PREFIX, object_name)
        return True
    except S3Error as e:
        logger.error(f"MinIO delete failed for {object_name}: {e}")
        return False


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _put_object(
    object_name: str,
    data: bytes,
    content_type: str,
    size: int,
) -> Optional[str]:
    try:
        client = get_client()
        client.put_object(
            BUCKET_PREFIX,
            object_name,
            io.BytesIO(data),
            length=size,
            content_type=content_type,
        )
        return f"{BUCKET_PREFIX}/{object_name}"
    except S3Error as e:
        logger.error(f"MinIO put failed for {object_name}: {e}")
        return None