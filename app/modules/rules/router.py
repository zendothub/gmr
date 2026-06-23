"""Rule API routes."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, get_current_user
from app.core.db.models.user import User
from app.modules.rules.schemas import RuleCreate, RuleUpdate, RuleResponse
from app.modules.rules.service import RuleService

router = APIRouter(prefix="/api/rules", tags=["Rules"])


@router.post("", response_model=RuleResponse, status_code=201)
async def create_rule(
    data: RuleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new rule."""
    rule = await RuleService.create_rule(db, data)
    return RuleResponse.model_validate(rule)


@router.get("", response_model=List[RuleResponse])
async def list_rules(
    camera_id: Optional[UUID] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all rules."""
    rules = await RuleService.get_rules(db, camera_id=camera_id)
    return [RuleResponse.model_validate(r) for r in rules]


@router.get("/{rule_id}", response_model=RuleResponse)
async def get_rule(
    rule_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a rule by ID."""
    rule = await RuleService.get_rule(db, rule_id)
    return RuleResponse.model_validate(rule)


@router.put("/{rule_id}", response_model=RuleResponse)
async def update_rule(
    rule_id: UUID,
    data: RuleUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update a rule."""
    rule = await RuleService.update_rule(db, rule_id, data)
    return RuleResponse.model_validate(rule)


@router.delete("/{rule_id}")
async def delete_rule(
    rule_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a rule."""
    return await RuleService.delete_rule(db, rule_id)


@router.post("/{rule_id}/enable", response_model=RuleResponse)
async def enable_rule(
    rule_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Enable a rule."""
    rule = await RuleService.enable_rule(db, rule_id)
    return RuleResponse.model_validate(rule)


@router.post("/{rule_id}/disable", response_model=RuleResponse)
async def disable_rule(
    rule_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Disable a rule."""
    rule = await RuleService.disable_rule(db, rule_id)
    return RuleResponse.model_validate(rule)