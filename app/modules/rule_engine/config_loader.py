"""Config loader - loads runtime configuration (rules, zones, views) from PostgreSQL.

This keeps the AI runtime decoupled from per-frame DB queries: configuration is
loaded once into memory and only refreshed when /api/runtime/reload-config is called.
"""

import uuid
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.camera import Camera, CameraView, Zone
from app.core.db.models.rule import Rule


def _enum_value(v):
    """Return .value for enums, pass through plain values."""
    return v.value if hasattr(v, "value") else v


async def load_camera_config(db: AsyncSession, camera_id: uuid.UUID) -> Optional[dict]:
    """Load a single camera config as a plain dict."""
    result = await db.execute(select(Camera).where(Camera.id == camera_id))
    camera = result.scalar_one_or_none()
    if not camera:
        return None
    return {
        "id": camera.id,
        "name": camera.name,
        "rtsp_url": camera.rtsp_url,
        "role": _enum_value(camera.role),
        "status": _enum_value(camera.status),
        "fps_target": camera.fps_target,
        "resolution": camera.resolution,
        "detection_model": camera.detection_model,
        "reid_enabled": camera.reid_enabled,
        "demographic_enabled": camera.demographic_enabled,
        "store_id": camera.store_id,
    }


async def load_active_cameras(db: AsyncSession) -> List[dict]:
    """Load all cameras marked as active."""
    result = await db.execute(select(Camera).where(Camera.is_active.is_(True)))
    cameras = result.scalars().all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "rtsp_url": c.rtsp_url,
            "role": _enum_value(c.role),
            "status": _enum_value(c.status),
            "fps_target": c.fps_target,
            "detection_model": c.detection_model,
            "reid_enabled": c.reid_enabled,
        }
        for c in cameras
    ]


async def load_views_for_camera(db: AsyncSession, camera_id: uuid.UUID) -> List[dict]:
    """Load active camera views (ROI polygons) for a camera."""
    result = await db.execute(
        select(CameraView).where(
            CameraView.camera_id == camera_id,
            CameraView.is_active.is_(True),
        )
    )
    views = result.scalars().all()
    return [
        {
            "id": v.id,
            "name": v.name,
            "view_type": _enum_value(v.view_type),
            "polygon": v.polygon,
            "is_default": v.is_default,
        }
        for v in views
    ]


async def load_zones_for_camera(db: AsyncSession, camera_id: uuid.UUID) -> List[dict]:
    """Load active zones across all views of a camera."""
    result = await db.execute(
        select(Zone)
        .join(CameraView, Zone.camera_view_id == CameraView.id)
        .where(
            CameraView.camera_id == camera_id,
            Zone.is_active.is_(True),
            CameraView.is_active.is_(True),
        )
    )
    zones = result.scalars().all()
    return [
        {
            "id": z.id,
            "name": z.name,
            "zone_type": _enum_value(z.zone_type),
            "shape": _enum_value(z.shape),
            "polygon": z.polygon,
            "line_config": z.line_config,
            "camera_view_id": z.camera_view_id,
        }
        for z in zones
    ]


async def load_active_rules(db: AsyncSession, camera_id: Optional[uuid.UUID] = None) -> List[dict]:
    """Load enabled rules (optionally scoped to a camera) as plain dicts."""
    query = select(Rule).where(Rule.is_enabled.is_(True))
    if camera_id:
        query = query.where((Rule.camera_id == camera_id) | (Rule.camera_id.is_(None)))
    result = await db.execute(query)
    rules = result.scalars().all()
    # IDs are stringified: the rule evaluator and zone cache work with string keys
    return [
        {
            "id": str(r.id),
            "name": r.name,
            "rule_type": _enum_value(r.rule_type),
            "zone_id": str(r.zone_id) if r.zone_id else None,
            "camera_id": str(r.camera_id) if r.camera_id else None,
            "config": r.config or {},
            "cooldown_seconds": r.cooldown_seconds,
            "severity": r.severity,
            "dwell_threshold_seconds": r.dwell_threshold_seconds,
            "count_threshold": r.count_threshold,
            "is_enabled": r.is_enabled,
        }
        for r in rules
    ]


async def load_runtime_config(db: AsyncSession, camera_id: uuid.UUID) -> Dict:
    """Load the full runtime configuration for a camera worker."""
    camera = await load_camera_config(db, camera_id)
    views = await load_views_for_camera(db, camera_id)
    zones = await load_zones_for_camera(db, camera_id)
    rules = await load_active_rules(db, camera_id)

    zones_by_id = {str(z["id"]): z for z in zones}

    logger.info(
        f"Runtime config loaded for camera {camera_id}: "
        f"{len(views)} views, {len(zones)} zones, {len(rules)} rules"
    )
    return {
        "camera": camera,
        "views": views,
        "zones": zones,
        "zones_by_id": zones_by_id,
        "rules": rules,
    }