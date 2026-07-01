"""Debug detection API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.debug.schemas import DebugListResponse
from app.modules.debug.service import DebugService


router = APIRouter(prefix="/api/v2/debug", tags=["Debug"])


@router.get("/detections", response_model=DebugListResponse)
async def get_debug_detections(
    store_id: Optional[UUID] = Query(None),
    camera_id: Optional[UUID] = Query(None),
    time_range: str = Query("today", pattern="^(today|this_week|custom)$"),
    start_time: Optional[datetime] = Query(None, description="Only used when time_range='custom'"),
    end_time: Optional[datetime] = Query(None, description="Only used when time_range='custom'"),
    status: str = Query("all", pattern="^(all|detected|not_detected)$"),
    page: int = Query(1, ge=1),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get detection debug records with filters.
    
    **Filters:**
    - `store_id`: Filter by store
    - `camera_id`: Filter by camera
    - `start_time`, `end_time`: Time range (default: last 24h)
    - `status`: all | detected | not_detected
    - `page`, `limit`: Pagination
    
    **Returns:**
    - Summary with detection rate and failure breakdown
    - Paginated list of debug records with all metrics
    """
    return await DebugService.get_debug_records(
        db=db,
        store_id=store_id,
        camera_id=camera_id,
        time_range=time_range,
        start_time=start_time,
        end_time=end_time,
        status=status,
        page=page,
        limit=limit,
    )
