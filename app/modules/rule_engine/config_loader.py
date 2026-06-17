"""Config loader - loads runtime configuration (rules, zones) from PostgreSQL.

This keeps the AI runtime decoupled from per-frame DB queries: configuration is
loaded once into memory and only refreshed when /api/runtime/reload-config is called.

Note: cameras are static (fixed mounting), so there is no per-camera "view"/ROI
concept anymore - detections run on the full frame and are filtered by zones.
"""

import uuid
from typing import Dict, List, Optional

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.camera import Camera, Zone
from app.core.db.models.rule import Rule


def _enum_value(v):
    """Return .value for enums, pass through plain values."""
    return v.value if hasattr(v, "value") else v


def _line_config_from_polygon(polygon: Optional[dict], shape: Optional[str]) -> Optional[dict]:
    """Derive a crossing line from the first two drawn points of a line-shaped zone.

    Returns ``{"start": [x1, y1], "end": [x2, y2]}`` or None when the zone is not a
    line / not enough points were drawn.
    """
    if shape != "line" or not polygon:
        return None
    points = polygon.get("points") if isinstance(polygon, dict) else None
    if not points or len(points) < 2:
        return None
    return {"start": points[0], "end": points[1]}


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
        "status": _enum_value(camera.status),
        "fps_target": camera.fps_target,
        "resolution": camera.resolution,
        "detection_model": camera.detection_model,
        "reid_enabled": camera.reid_enabled,
        "demographic_enabled": camera.demographic_enabled,
        "frame_rotation": camera.frame_rotation,
        "burnin_enabled": bool(camera.burnin_enabled),
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
            "status": _enum_value(c.status),
            "fps_target": c.fps_target,
            "detection_model": c.detection_model,
            "reid_enabled": c.reid_enabled,
        }
        for c in cameras
    ]


async def load_zones_for_camera(db: AsyncSession, camera_id: uuid.UUID) -> List[dict]:
    """Load all active zones bound to this camera (camera -> many zones)."""
    result = await db.execute(
        select(Zone).where(
            Zone.camera_id == camera_id,
            Zone.is_active.is_(True),
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
            # Line-crossing config is derived from the drawn polygon points so the
            # operator only ever draws one shape (no separate line_config field).
            "line_config": _line_config_from_polygon(z.polygon, _enum_value(z.shape)),
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
    zones = await load_zones_for_camera(db, camera_id)
    rules = await load_active_rules(db, camera_id)

    zones_by_id = {str(z["id"]): z for z in zones}

    logger.info(
        f"Runtime config loaded for camera {camera_id}: "
        f"{len(zones)} zones, {len(rules)} rules"
    )
    return {
        "camera": camera,
        "zones": zones,
        "zones_by_id": zones_by_id,
        "rules": rules,
    }
