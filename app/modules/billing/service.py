"""Billing interaction service."""

from datetime import datetime
from typing import Optional, Tuple, List
from uuid import UUID

from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.billing import BillingInteraction
from app.utils.time_utils import utc_now


class BillingService:

    @staticmethod
    async def get_interactions(
        db: AsyncSession,
        camera_id: Optional[UUID] = None,
        person_identity_id: Optional[UUID] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[BillingInteraction], int]:
        """List billing interactions with filters and pagination."""
        query = select(BillingInteraction)

        if camera_id:
            query = query.where(BillingInteraction.camera_id == camera_id)
        if person_identity_id:
            query = query.where(BillingInteraction.person_identity_id == person_identity_id)
        if start_time:
            query = query.where(BillingInteraction.entered_at >= start_time)
        if end_time:
            query = query.where(BillingInteraction.entered_at <= end_time)

        count_query = select(func.count()).select_from(query.subquery())
        total = (await db.execute(count_query)).scalar() or 0

        query = (
            query.order_by(BillingInteraction.entered_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(query)
        return list(result.scalars().all()), total

    @staticmethod
    async def get_interaction(db: AsyncSession, interaction_id: UUID) -> BillingInteraction:
        """Get a single billing interaction."""
        result = await db.execute(
            select(BillingInteraction).where(BillingInteraction.id == interaction_id)
        )
        interaction = result.scalar_one_or_none()
        if not interaction:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Billing interaction not found")
        return interaction

    @staticmethod
    async def open_interaction(
        db: AsyncSession,
        camera_id: UUID,
        zone_id: Optional[UUID] = None,
        person_identity_id: Optional[UUID] = None,
        track_session_id: Optional[UUID] = None,
        metadata: Optional[dict] = None,
    ) -> BillingInteraction:
        """Record a person entering the billing zone (used by the AI runtime)."""
        interaction = BillingInteraction(
            camera_id=camera_id,
            zone_id=zone_id,
            person_identity_id=person_identity_id,
            track_session_id=track_session_id,
            entered_at=utc_now(),
            interaction_type="billing_counter",
            metadata_json=metadata,
        )
        db.add(interaction)
        await db.flush()
        logger.info(f"Billing interaction opened: {interaction.id} (camera={camera_id})")
        return interaction

    @staticmethod
    async def close_interaction(
        db: AsyncSession, interaction_id: UUID
    ) -> BillingInteraction:
        """Record the person leaving the billing zone and compute dwell."""
        interaction = await BillingService.get_interaction(db, interaction_id)
        interaction.exited_at = utc_now()
        if interaction.entered_at:
            interaction.dwell_seconds = (
                interaction.exited_at - interaction.entered_at
            ).total_seconds()
        await db.flush()
        logger.info(
            f"Billing interaction closed: {interaction_id} "
            f"dwell={interaction.dwell_seconds}s"
        )
        return interaction