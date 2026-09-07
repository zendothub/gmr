"""Pydantic schemas for ShiftSlot CRUD."""

from datetime import time
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, field_validator, model_validator


class ShiftSlotCreate(BaseModel):
    label: str
    start_time: time   # "11:00" or "23:00"
    end_time: time     # "19:00" or "04:00"

    @model_validator(mode="after")
    def set_crosses_midnight(self) -> "ShiftSlotCreate":
        # crosses_midnight is derived — just keep it in the schema for clarity
        return self


class ShiftSlotUpdate(BaseModel):
    label: Optional[str] = None
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    is_active: Optional[bool] = None


class ShiftSlotResponse(BaseModel):
    id: UUID
    label: str
    start_time: time
    end_time: time
    crosses_midnight: bool
    is_active: bool

    model_config = {"from_attributes": True}
