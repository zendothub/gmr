"""Storage service — MinIO (S3-compatible object storage).

All binary objects (snapshots, crops, clips, reports) are stored exclusively
in MinIO.  The PostgreSQL ``storage_objects`` table tracks metadata regardless.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, Tuple, List
from io import BytesIO

import numpy as np
from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.db.models.storage import StorageObject, StorageType
from app.utils.time_utils import utc_now

settings = get_settings()


# ---------------------------------------------------------------------------
# Object-name helpers
# ---------------------------------------------------------------------------

def _object_name(storage_type: StorageType, prefix: str = "img") -> str:
    """Build a MinIO object key for a new blob."""
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    uid = uuid.uuid4().hex[:8]
    return f"{storage_type.value}s/{prefix}_{ts}_{uid}.jpg"


def _object_key(storage_type: StorageType, object_name: str) -> str:
    """Extract just the object key (strip bucket prefix if present)."""
    if object_name.startswith(f"{settings.MINIO_BUCKET_PREFIX}/"):
        return object_name[len(settings.MINIO_BUCKET_PREFIX) + 1:]
    return object_name


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

async def save_image_bytes(
    image: np.ndarray,
    storage_type: StorageType,
    prefix: str = "img",
    quality: int = 85,
) -> Optional[bytes]:
    """Encode a numpy image to JPEG bytes (returned, *not* persisted here).

    Callers use this + ``register_and_upload`` to persist.
    """
    try:
        import cv2
        _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
    except Exception as e:
        logger.error(f"Failed to encode image: {e}")
        return None


async def upload_to_storage(
    data: bytes,
    storage_type: StorageType,
    object_name: str,
    content_type: str = "image/jpeg",
) -> Optional[str]:
    """Persist bytes to MinIO.

    Returns the object key on success.
    """
    from app.modules.storage.minio_client import upload_bytes

    result = upload_bytes(object_name, data, content_type)
    if result is None:
        logger.error(f"MinIO upload failed for {object_name}")
        return None
    return object_name


async def register_and_upload(
    db: AsyncSession,
    image: np.ndarray,
    storage_type: StorageType,
    camera_id: Optional[uuid.UUID] = None,
    event_id: Optional[uuid.UUID] = None,
    person_identity_id: Optional[uuid.UUID] = None,
    prefix: str = "img",
) -> Optional[StorageObject]:
    """One-shot: encode image, upload, and register the metadata row.

    This is the main entry point used by camera workers and event handlers.
    """
    obj_name = _object_name(storage_type, prefix)
    data = await save_image_bytes(image, storage_type, prefix)
    if data is None:
        return None

    path = await upload_to_storage(data, storage_type, obj_name)
    if path is None:
        return None

    obj = StorageObject(
        file_path=path,
        file_name=obj_name.rsplit("/", 1)[-1],
        storage_type=storage_type,
        mime_type="image/jpeg",
        file_size_bytes=len(data),
        camera_id=camera_id,
        event_id=event_id,
        person_identity_id=person_identity_id,
        captured_at=utc_now(),
    )
    db.add(obj)
    await db.flush()
    logger.debug(f"Storage object registered: {obj.file_path}")
    return obj


# ---------------------------------------------------------------------------
# Download / serve
# ---------------------------------------------------------------------------

async def download_object(db: AsyncSession, object_id: uuid.UUID) -> Tuple[bytes, str, str]:
    """Return (bytes, filename, mime_type) for a stored object."""
    result = await db.execute(select(StorageObject).where(StorageObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Storage object not found")

    from app.modules.storage.minio_client import download_bytes as minio_download

    data = minio_download(obj.file_path)
    if data is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "File not found in MinIO")

    return data, obj.file_name, obj.mime_type or "application/octet-stream"


async def stream_object_response(db: AsyncSession, object_id: uuid.UUID) -> StreamingResponse:
    """Return a FastAPI ``StreamingResponse`` for a stored object."""
    data, filename, mime = await download_object(db, object_id)
    return StreamingResponse(
        BytesIO(data),
        media_type=mime,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

async def delete_storage_object(db: AsyncSession, obj: StorageObject) -> None:
    """Remove a storage object from MinIO and its DB row."""
    from app.modules.storage.minio_client import delete_object as minio_delete

    minio_delete(obj.file_path)
    await db.delete(obj)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

async def cleanup_old_objects(db: AsyncSession, older_than: datetime) -> int:
    """Purge old storage objects (blob + DB row). Returns count removed."""
    from app.modules.storage.minio_client import delete_object as minio_delete

    result = await db.execute(
        select(StorageObject).where(StorageObject.created_at < older_than)
    )
    objects = list(result.scalars().all())

    removed = 0
    for obj in objects:
        try:
            minio_delete(obj.file_path)
            await db.delete(obj)
            removed += 1
        except Exception as e:
            logger.error(f"Failed to remove MinIO object {obj.id}: {e}")

    logger.info(f"Storage cleanup removed {removed} objects older than {older_than}")
    return removed


# ---------------------------------------------------------------------------
# List (unchanged — metadata-only, no backend call needed)
# ---------------------------------------------------------------------------

async def list_objects(
    db: AsyncSession,
    storage_type: Optional[str] = None,
    camera_id: Optional[uuid.UUID] = None,
    event_id: Optional[uuid.UUID] = None,
    page: int = 1,
    page_size: int = 50,
) -> Tuple[List[StorageObject], int]:
    query = select(StorageObject)
    if storage_type:
        query = query.where(StorageObject.storage_type == storage_type)
    if camera_id is not None:
        query = query.where(StorageObject.camera_id == camera_id)
    if event_id is not None:
        query = query.where(StorageObject.event_id == event_id)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    query = (
        query.order_by(StorageObject.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(query)
    return list(result.scalars().all()), total


async def get_object(db: AsyncSession, object_id: uuid.UUID) -> StorageObject:
    result = await db.execute(select(StorageObject).where(StorageObject.id == object_id))
    obj = result.scalar_one_or_none()
    if not obj:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Storage object not found")
    return obj