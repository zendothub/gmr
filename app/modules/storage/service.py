"""Storage service - local filesystem storage for snapshots, crops, clips, reports.

Clean module boundary: if object storage (e.g. MinIO/S3) is needed later,
only this module needs to change.
"""

import os
import uuid
from datetime import datetime
from typing import Optional, Tuple, List
from uuid import UUID

import numpy as np
from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.db.models.storage import StorageObject, StorageType
from app.utils.image_utils import save_image
from app.utils.time_utils import utc_now


class StorageService:
    """Local filesystem storage with PostgreSQL metadata registry."""

    def __init__(self):
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def get_dir(self, storage_type: StorageType) -> str:
        """Resolve the directory for a storage type."""
        mapping = {
            StorageType.SNAPSHOT: self.settings.SNAPSHOT_DIR,
            StorageType.CROP: self.settings.CROP_DIR,
            StorageType.CLIP: self.settings.CLIP_DIR,
            StorageType.REPORT: self.settings.REPORT_DIR,
        }
        directory = os.path.join(self.settings.STORAGE_ROOT, mapping[storage_type])
        os.makedirs(directory, exist_ok=True)
        return directory

    # ------------------------------------------------------------------
    # Save + register
    # ------------------------------------------------------------------

    async def save_image_object(
        self,
        db: AsyncSession,
        image: np.ndarray,
        storage_type: StorageType,
        camera_id: Optional[UUID] = None,
        event_id: Optional[UUID] = None,
        person_identity_id: Optional[UUID] = None,
        prefix: str = "img",
    ) -> Optional[StorageObject]:
        """Save an image to local storage and register it in storage_objects."""
        directory = self.get_dir(storage_type)
        file_path = save_image(image, directory, prefix=prefix)
        if not file_path:
            logger.error(f"Failed to save image to {directory}")
            return None

        return await self.register_object(
            db,
            file_path=file_path,
            storage_type=storage_type,
            camera_id=camera_id,
            event_id=event_id,
            person_identity_id=person_identity_id,
            mime_type="image/jpeg",
        )

    async def register_object(
        self,
        db: AsyncSession,
        file_path: str,
        storage_type: StorageType,
        camera_id: Optional[UUID] = None,
        event_id: Optional[UUID] = None,
        person_identity_id: Optional[UUID] = None,
        mime_type: Optional[str] = None,
    ) -> StorageObject:
        """Register an existing file in the storage_objects table."""
        file_size = None
        try:
            file_size = os.path.getsize(file_path)
        except OSError:
            pass

        obj = StorageObject(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            storage_type=storage_type,
            mime_type=mime_type,
            file_size_bytes=file_size,
            camera_id=camera_id,
            event_id=event_id,
            person_identity_id=person_identity_id,
            captured_at=utc_now(),
        )
        db.add(obj)
        await db.flush()
        logger.debug(f"Storage object registered: {obj.file_path}")
        return obj

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @staticmethod
    async def list_objects(
        db: AsyncSession,
        storage_type: Optional[str] = None,
        camera_id: Optional[UUID] = None,
        event_id: Optional[UUID] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[StorageObject], int]:
        """List storage objects with filters."""
        query = select(StorageObject)
        if storage_type:
            query = query.where(StorageObject.storage_type == storage_type)
        if camera_id:
            query = query.where(StorageObject.camera_id == camera_id)
        if event_id:
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

    @staticmethod
    async def get_object(db: AsyncSession, object_id: UUID) -> StorageObject:
        """Get a storage object by ID."""
        result = await db.execute(
            select(StorageObject).where(StorageObject.id == object_id)
        )
        obj = result.scalar_one_or_none()
        if not obj:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Storage object not found")
        return obj

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    async def cleanup_old_objects(db: AsyncSession, older_than: datetime) -> int:
        """Delete old storage objects (files + DB rows). Returns removed count."""
        result = await db.execute(
            select(StorageObject).where(StorageObject.created_at < older_than)
        )
        objects = list(result.scalars().all())

        removed = 0
        for obj in objects:
            try:
                if os.path.exists(obj.file_path):
                    os.remove(obj.file_path)
                await db.delete(obj)
                removed += 1
            except Exception as e:
                logger.error(f"Failed to remove storage object {obj.id}: {e}")

        logger.info(f"Storage cleanup removed {removed} objects older than {older_than}")
        return removed


storage_service = StorageService()