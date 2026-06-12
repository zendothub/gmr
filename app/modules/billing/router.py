"""Billing API routes (dashboard integration)."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.billing.schemas import (
    BillingInteractionResponse,
    BillingInteractionListResponse,
)
from app.modules.billing.service import BillingService

router = APIRouter(prefix="/api/billing", tags=["Billing"])


@router.get("/interactions", response_model=BillingInteractionListResponse)
async def list_billing_interactions(
    camera_id: Optional[UUID] = Query(None),
    person_identity_id: Optional[UUID] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List billing interactions with filters and pagination."""
    items, total = await BillingService.get_interactions(
        db,
        camera_id=camera_id,
        person_identity_id=person_identity_id,
        start_time=start_time,
        end_time=end_time,
        page=page,
        page_size=page_size,
    )
    return BillingInteractionListResponse(
        items=[BillingInteractionResponse.model_validate(i) for i in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/interactions/{interaction_id}", response_model=BillingInteractionResponse)
async def get_billing_interaction(
    interaction_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single billing interaction by ID."""
    interaction = await BillingService.get_interaction(db, interaction_id)
    return BillingInteractionResponse.model_validate(interaction)