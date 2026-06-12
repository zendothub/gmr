"""Storage API routes - browse and download stored media."""

import os
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.storage.service import StorageService

router = APIRouter(prefix="/api/storage", tags=["Storage"])


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
    """List stored media objects."""
    items, total = await StorageService.list_objects(
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
    """Download a stored media file."""
    obj = await StorageService.get_object(db, object_id)
    if not os.path.exists(obj.file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    return FileResponse(
        path=obj.file_path,
        filename=obj.file_name,
        media_type=obj.mime_type or "application/octet-stream",
    )