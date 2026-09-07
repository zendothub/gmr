"""ShiftSlot CRUD service."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import HTTPException

from app.core.db.models.attendance import ShiftSlot
from app.modules.shift_slots.schemas import ShiftSlotCreate, ShiftSlotUpdate


class ShiftSlotService:

    @staticmethod
    async def create(db: AsyncSession, payload: ShiftSlotCreate) -> ShiftSlot:
        """Create a new shift slot. Label must be unique."""
        # Duplicate label check
        existing = (
            await db.execute(select(ShiftSlot).where(ShiftSlot.label == payload.label))
        ).scalar_one_or_none()
        if existing:
            raise HTTPException(status_code=409, detail=f"Shift slot '{payload.label}' already exists.")

        crosses_midnight = payload.end_time < payload.start_time

        slot = ShiftSlot(
            label=payload.label,
            start_time=payload.start_time,
            end_time=payload.end_time,
            crosses_midnight=crosses_midnight,
        )
        db.add(slot)
        await db.commit()
        await db.refresh(slot)
        return slot

    @staticmethod
    async def list_all(db: AsyncSession, active_only: bool = True) -> List[ShiftSlot]:
        q = select(ShiftSlot)
        if active_only:
            q = q.where(ShiftSlot.is_active.is_(True))
        q = q.order_by(ShiftSlot.start_time)
        result = await db.execute(q)
        return result.scalars().all()

    @staticmethod
    async def get_by_id(db: AsyncSession, slot_id: UUID) -> ShiftSlot:
        slot = (
            await db.execute(select(ShiftSlot).where(ShiftSlot.id == slot_id))
        ).scalar_one_or_none()
        if not slot:
            raise HTTPException(status_code=404, detail="Shift slot not found.")
        return slot

    @staticmethod
    async def update(db: AsyncSession, slot_id: UUID, payload: ShiftSlotUpdate) -> ShiftSlot:
        slot = await ShiftSlotService.get_by_id(db, slot_id)

        if payload.label is not None:
            # Check uniqueness if label is changing
            if payload.label != slot.label:
                dup = (
                    await db.execute(select(ShiftSlot).where(ShiftSlot.label == payload.label))
                ).scalar_one_or_none()
                if dup:
                    raise HTTPException(status_code=409, detail=f"Shift slot '{payload.label}' already exists.")
            slot.label = payload.label

        if payload.start_time is not None:
            slot.start_time = payload.start_time
        if payload.end_time is not None:
            slot.end_time = payload.end_time
        if payload.is_active is not None:
            slot.is_active = payload.is_active

        # Recompute crosses_midnight
        slot.crosses_midnight = slot.end_time < slot.start_time

        await db.commit()
        await db.refresh(slot)
        return slot

    @staticmethod
    async def delete(db: AsyncSession, slot_id: UUID) -> None:
        """Soft-delete by marking is_active=False (preserves FK in attendance_records)."""
        slot = await ShiftSlotService.get_by_id(db, slot_id)
        slot.is_active = False
        await db.commit()
