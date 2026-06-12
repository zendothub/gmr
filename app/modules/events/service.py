"""Events service - query and manage detection events."""

from datetime import datetime
from typing import Optional, Tuple, List
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.event import Event


class EventService:

    @staticmethod
    async def get_events(
        db: AsyncSession,
        camera_id: Optional[UUID] = None,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        person_identity_id: Optional[UUID] = None,
        is_acknowledged: Optional[bool] = None,
        include_false_positives: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[Event], int]:
        """List events with filters and pagination. Returns (items, total)."""
        query = select(Event)

        if camera_id:
            query = query.where(Event.camera_id == camera_id)
        if event_type:
            query = query.where(Event.event_type == event_type)
        if severity:
            query = query.where(Event.severity == severity)
        if person_identity_id:
            query = query.where(Event.person_identity_id == person_identity_id)
        if is_acknowledged is not None:
            query = query.where(Event.is_acknowledged == is_acknowledged)
        if not include_false_positives:
            query = query.where(Event.is_false_positive.is_(False))
        if start_time:
            query = query.where(Event.occurred_at >= start_time)
        if end_time:
            query = query.where(Event.occurred_at <= end_time)

        # Total count
        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar() or 0

        # Page
        query = (
            query.order_by(Event.occurred_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(query)
        return list(result.scalars().all()), total

    @staticmethod
    async def get_event(db: AsyncSession, event_id: UUID) -> Event:
        """Get a single event by ID."""
        result = await db.execute(select(Event).where(Event.id == event_id))
        event = result.scalar_one_or_none()
        if not event:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Event not found")
        return event

    @staticmethod
    async def acknowledge_event(db: AsyncSession, event_id: UUID, user_id: UUID) -> Event:
        """Mark an event as acknowledged."""
        event = await EventService.get_event(db, event_id)
        event.is_acknowledged = True
        event.acknowledged_by = user_id
        await db.flush()
        await db.refresh(event)
        logger.info(f"Event acknowledged: {event_id} by user {user_id}")
        return event

    @staticmethod
    async def mark_false_positive(db: AsyncSession, event_id: UUID, user_id: UUID) -> Event:
        """Mark an event as a false positive (also acknowledges it)."""
        event = await EventService.get_event(db, event_id)
        event.is_false_positive = True
        event.is_acknowledged = True
        event.acknowledged_by = user_id
        await db.flush()
        await db.refresh(event)
        logger.info(f"Event marked as false positive: {event_id} by user {user_id}")
        return event