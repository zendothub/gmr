"""AI Runtime API routes - control the camera worker fleet."""

from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_current_user, require_role
from app.core.db.models.user import User
from app.modules.ai_runtime.worker_supervisor import WorkerSupervisor

router = APIRouter(prefix="/api/runtime", tags=["AI Runtime"])


@router.post("/reload-config")
async def reload_config(current_user: User = Depends(require_role("admin"))):
    """Reload rules/zones/views from PostgreSQL into all running workers (admin only).

    The rule engine never queries the DB per frame - it works from an
    in-memory cache that is refreshed only via this endpoint.
    """
    supervisor = WorkerSupervisor.get_instance()
    result = await supervisor.reload_config()
    return {"message": "Runtime configuration reloaded", **result}


@router.post("/start")
async def start_runtime(current_user: User = Depends(require_role("admin"))):
    """Start workers for all cameras marked active in the database (admin only)."""
    supervisor = WorkerSupervisor.get_instance()
    try:
        result = await supervisor.start_all_active()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start runtime: {e}")
    return {"message": "Runtime start requested", **result}


@router.post("/stop")
async def stop_runtime(current_user: User = Depends(require_role("admin"))):
    """Stop all running camera workers (admin only)."""
    supervisor = WorkerSupervisor.get_instance()
    stopped = await supervisor.stop_all()
    return {"message": "Runtime stopped", "stopped_workers": stopped}


@router.get("/status")
async def runtime_status(current_user: User = Depends(get_current_user)):
    """Get status of all camera workers."""
    supervisor = WorkerSupervisor.get_instance()
    return supervisor.get_all_status()