"""Read-only recording status (no DB persistence)."""

from fastapi import APIRouter

from app.modules.recording.service import RecordingSupervisor

router = APIRouter(prefix="/api/recording", tags=["Recording"])


@router.get("/status")
async def recording_status():
    return RecordingSupervisor.get_instance().status()
