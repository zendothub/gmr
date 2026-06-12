"""Camera View service - CRUD for ROI/view selection."""

from typing import List
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.camera import CameraView
from app.modules.camera_views.schemas import CameraViewCreate, CameraViewUpdate


class CameraViewService:

    @staticmethod
    async def create_view(db: AsyncSession, camera_id: UUID, data: CameraViewCreate) -> CameraView:
        """Create a new camera view/ROI for a camera."""
        view = CameraView(
            camera_id=camera_id,
            name=data.name,
            view_type=data.view_type,
            polygon=data.polygon,
            is_default=data.is_default,
            is_active=data.is_active,
        )
        # If this is set as default, unset others
        if data.is_default:
            await db.execute(
                update(CameraView)
                .where(CameraView.camera_id == camera_id, CameraView.is_default == True)
                .values(is_default=False)
            )

        db.add(view)
        await db.flush()
        await db.refresh(view)
        logger.info(f"Camera view created: {view.name} for camera {camera_id}")
        return view

    @staticmethod
    async def get_views_for_camera(db: AsyncSession, camera_id: UUID) -> List[CameraView]:
        """Get all views for a camera."""
        result = await db.execute(
            select(CameraView)
            .where(CameraView.camera_id == camera_id)
            .order_by(CameraView.created_at)
        )
        return list(result.scalars().all())

    @staticmethod
    async def get_view(db: AsyncSession, view_id: UUID) -> CameraView:
        """Get a camera view by ID."""
        result = await db.execute(select(CameraView).where(CameraView.id == view_id))
        view = result.scalar_one_or_none()
        if not view:
            raise HTTPException(status_code=404, detail="Camera view not found")
        return view

    @staticmethod
    async def update_view(db: AsyncSession, view_id: UUID, data: CameraViewUpdate) -> CameraView:
        """Update a camera view."""
        view = await CameraViewService.get_view(db, view_id)
        update_data = data.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            setattr(view, key, value)

        await db.flush()
        await db.refresh(view)
        logger.info(f"Camera view updated: {view.name} (id={view_id})")
        return view

    @staticmethod
    async def delete_view(db: AsyncSession, view_id: UUID) -> dict:
        """Delete a camera view."""
        view = await CameraViewService.get_view(db, view_id)
        await db.delete(view)
        logger.info(f"Camera view deleted: {view.name} (id={view_id})")
        return {"message": f"Camera view '{view.name}' deleted successfully"}

    @staticmethod
    async def set_default_view(db: AsyncSession, view_id: UUID) -> CameraView:
        """Set a camera view as default, unsetting others for that camera."""
        view = await CameraViewService.get_view(db, view_id)
        # Unset all other defaults for this camera
        await db.execute(
            update(CameraView)
            .where(CameraView.camera_id == view.camera_id, CameraView.is_default == True)
            .values(is_default=False)
        )
        view.is_default = True
        await db.flush()
        await db.refresh(view)
        logger.info(f"Default view set: {view.name} for camera {view.camera_id}")
        return view
