"""Storage API routes — browse and download stored media.

Now routes through MinIO (when configured) or local filesystem (fallback).
"""

from datetime import timedelta
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.storage import service as storage_svc
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX

router = APIRouter(prefix="/api/storage", tags=["Storage"])


@router.get("/presigned-url")
async def get_presigned_url(
    path: str = Query(..., description="MinIO object path (e.g., retail/crops/crop_xxx.jpg)"),
    expires: int = Query(3600, ge=60, le=86400, description="Expiry in seconds (default 1h)"),
    current_user: User = Depends(get_current_user),
):
    """Return a presigned GET URL for a MinIO object.

    The browser uses the returned URL to display images directly from MinIO.
    Path format: {bucket}/{object_name} (e.g., retail/crops/crop_abc.jpg).
    """
    try:
        client = get_client()
        # path includes bucket prefix, split it out
        if "/" in path:
            bucket, object_name = path.split("/", 1)
        else:
            bucket = BUCKET_PREFIX
            object_name = path

        url = client.presigned_get_object(
            bucket_name=bucket,
            object_name=object_name,
            expires=timedelta(seconds=expires),
        )
        return {"url": url, "expires_in": expires}
    except Exception as exc:
        logger.warning(f"presigned-url failed: bucket={bucket} object={object_name} error={exc}")
        raise HTTPException(status_code=400, detail=f"Failed to generate presigned URL: {exc}")


@router.get("/objects")
async def list_storage_objects(
    storage_type: Optional[str] = Query(None, pattern="^(snapshot|crop|clip|report)$"),
    camera_id: Optional[UUID] = Query(None),
    event_id: Optional[UUID] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List stored media objects (metadata from Postgres, blobs in MinIO/local)."""
    items, total = await storage_svc.list_objects(
        db,
        storage_type=storage_type,
        camera_id=camera_id,
        event_id=event_id,
        page=page,
        page_size=page_size,
    )
    return {
        "items": [
            {
                "id": str(o.id),
                "file_name": o.file_name,
                "file_path": o.file_path,
                "storage_type": o.storage_type.value if hasattr(o.storage_type, "value") else o.storage_type,
                "mime_type": o.mime_type,
                "file_size_bytes": o.file_size_bytes,
                "camera_id": str(o.camera_id) if o.camera_id else None,
                "event_id": str(o.event_id) if o.event_id else None,
                "captured_at": o.captured_at.isoformat() if o.captured_at else None,
            }
            for o in items
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/objects/{object_id}/download")
async def download_storage_object(
    object_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download a stored media file (streamed from MinIO or local disk)."""
    return await storage_svc.stream_object_response(db, object_id)