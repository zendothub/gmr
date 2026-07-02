"""Debug detection API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.debug.schemas import ActiveTracksRealtimeResponse
from app.modules.debug.service import DebugService


router = APIRouter(prefix="/api/v2/debug", tags=["Debug"])


@router.get("/active-tracks", response_model=ActiveTracksRealtimeResponse)
async def get_active_tracks(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Get all active tracks currently in memory across all cameras.
    
    **Returns:**
    - Summary of active, identified, and inactive track counts
    - List of active tracks with current quality, face status, and ReID scores
    """
    return await DebugService.get_active_tracks(db)
