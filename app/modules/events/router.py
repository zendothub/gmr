"""Events API routes."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.events.schemas import EventResponse, EventListResponse
from app.modules.events.service import EventService

router = APIRouter(prefix="/api/alerts", tags=["Alerts"])


@router.get("", response_model=EventListResponse)
async def list_events(
    camera_id: Optional[UUID] = Query(None),
    event_type: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    person_identity_id: Optional[UUID] = Query(None),
    is_acknowledged: Optional[bool] = Query(None),
    include_false_positives: bool = Query(False),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List events with filters and pagination."""
    items, total = await EventService.get_events(
        db,
        camera_id=camera_id,
        event_type=event_type,
        severity=severity,
        person_identity_id=person_identity_id,
        is_acknowledged=is_acknowledged,
        include_false_positives=include_false_positives,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )
    return EventListResponse(
        items=[EventResponse.model_validate(e) for e in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{alerts}", response_model=EventResponse)
async def get_event(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single event by ID."""
    event = await EventService.get_event(db, event_id)
    return EventResponse.model_validate(event)


@router.post("/{alert_id}/acknowledge", response_model=EventResponse)
async def acknowledge_event(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Acknowledge an event."""
    event = await EventService.acknowledge_event(db, event_id, current_user.id)
    return EventResponse.model_validate(event)


@router.post("/{alert_id}/false-positive", response_model=EventResponse)
async def mark_false_positive(
    event_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Mark an event as a false positive."""
    event = await EventService.mark_false_positive(db, event_id, current_user.id)
    return EventResponse.model_validate(event)