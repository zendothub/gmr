"""Rule service - CRUD and enable/disable."""

import asyncio
from typing import List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException
from loguru import logger

from app.core.db.models.rule import Rule
from app.modules.rules.schemas import RuleCreate, RuleUpdate


def _schedule_config_reload(camera_id: Optional[UUID] = None):
    """Fire-and-forget config reload for running workers.

    If *camera_id* is provided only that camera's worker is reloaded;
    otherwise all workers are refreshed.  The reload runs in the background
    so the API response is never delayed.
    """
    try:
        from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor
        supervisor = WorkerSupervisor.get_instance()
        loop = asyncio.get_running_loop()
        loop.create_task(supervisor.reload_config())
        logger.info(
            f"Scheduled runtime config reload "
            f"({'camera=' + str(camera_id) if camera_id else 'all workers'})"
        )
    except RuntimeError:
        # No running event loop (e.g. tests) — silently skip
        pass
    except Exception as exc:
        logger.debug(f"Could not schedule config reload: {exc}")


class RuleService:

    @staticmethod
    async def create_rule(db: AsyncSession, data: RuleCreate) -> Rule:
        """Create a new rule."""
        rule = Rule(
            name=data.name,
            rule_type=data.rule_type,
            zone_id=data.zone_id,
            camera_id=data.camera_id,
            config=data.config or {},
            cooldown_seconds=data.cooldown_seconds,
            severity=data.severity,
            dwell_threshold_seconds=data.dwell_threshold_seconds,
            count_threshold=data.count_threshold,
            is_enabled=data.is_enabled,
        )
        db.add(rule)
        await db.flush()
        await db.refresh(rule)
        logger.info(f"Rule created: {rule.name} (type={rule.rule_type})")
        _schedule_config_reload(data.camera_id)
        return rule

    @staticmethod
    async def get_rules(db: AsyncSession, camera_id: UUID = None, enabled_only: bool = False) -> List[Rule]:
        """List all rules, optionally filtered."""
        query = select(Rule)
        if camera_id:
            query = query.where(Rule.camera_id == camera_id)
        if enabled_only:
            query = query.where(Rule.is_enabled == True)
        query = query.order_by(Rule.created_at.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    @staticmethod
    async def get_rule(db: AsyncSession, rule_id: UUID) -> Rule:
        """Get a rule by ID."""
        result = await db.execute(select(Rule).where(Rule.id == rule_id))
        rule = result.scalar_one_or_none()
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        return rule

    @staticmethod
    async def update_rule(db: AsyncSession, rule_id: UUID, data: RuleUpdate) -> Rule:
        """Update a rule."""
        rule = await RuleService.get_rule(db, rule_id)
        update_data = data.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            setattr(rule, key, value)

        await db.flush()
        await db.refresh(rule)
        logger.info(f"Rule updated: {rule.name} (id={rule_id})")
        _schedule_config_reload(rule.camera_id)
        return rule

    @staticmethod
    async def delete_rule(db: AsyncSession, rule_id: UUID) -> dict:
        """Delete a rule."""
        rule = await RuleService.get_rule(db, rule_id)
        camera_id = rule.camera_id
        await db.delete(rule)
        logger.info(f"Rule deleted: {rule.name} (id={rule_id})")
        _schedule_config_reload(camera_id)
        return {"message": f"Rule '{rule.name}' deleted successfully"}

    @staticmethod
    async def enable_rule(db: AsyncSession, rule_id: UUID) -> Rule:
        """Enable a rule."""
        rule = await RuleService.get_rule(db, rule_id)
        rule.is_enabled = True
        await db.flush()
        await db.refresh(rule)
        logger.info(f"Rule enabled: {rule.name}")
        _schedule_config_reload(rule.camera_id)
        return rule

    @staticmethod
    async def disable_rule(db: AsyncSession, rule_id: UUID) -> Rule:
        """Disable a rule."""
        rule = await RuleService.get_rule(db, rule_id)
        rule.is_enabled = False
        await db.flush()
        await db.refresh(rule)
        logger.info(f"Rule disabled: {rule.name}")
        _schedule_config_reload(rule.camera_id)
        return rule
