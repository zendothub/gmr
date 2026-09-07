"""ShiftSlot CRUD API routes."""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.shift_slots.schemas import ShiftSlotCreate, ShiftSlotUpdate, ShiftSlotResponse
from app.modules.shift_slots.service import ShiftSlotService

router = APIRouter(prefix="/api/v1/shift-slots", tags=["Shift Slots"])


@router.post("", response_model=ShiftSlotResponse, status_code=201)
async def create_shift_slot(
    payload: ShiftSlotCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new shift slot.

    - **label**: human-readable name e.g. `"11AM-7PM"`
    - **start_time**: shift start in `HH:MM` format e.g. `"11:00"`
    - **end_time**: shift end in `HH:MM` format e.g. `"19:00"` (use `"04:00"` for night shifts ending after midnight)

    `crosses_midnight` is auto-computed — if `end_time < start_time` the shift spans two calendar days.
    """
    return await ShiftSlotService.create(db, payload)


@router.get("", response_model=List[ShiftSlotResponse])
async def list_shift_slots(
    active_only: bool = Query(True, description="Return only active slots"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all shift slots."""
    return await ShiftSlotService.list_all(db, active_only=active_only)


@router.get("/{slot_id}", response_model=ShiftSlotResponse)
async def get_shift_slot(
    slot_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single shift slot by ID."""
    return await ShiftSlotService.get_by_id(db, slot_id)


@router.put("/{slot_id}", response_model=ShiftSlotResponse)
async def update_shift_slot(
    slot_id: UUID,
    payload: ShiftSlotUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update label, times, or active status of a shift slot."""
    return await ShiftSlotService.update(db, slot_id, payload)


@router.delete("/{slot_id}", status_code=204)
async def delete_shift_slot(
    slot_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Soft-delete a shift slot (marks `is_active=False`).
    Existing attendance records that reference this slot are preserved.
    """
    await ShiftSlotService.delete(db, slot_id)
