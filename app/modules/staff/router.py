"""Staff registration API routes."""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db.models.user import User
from app.dependencies import get_current_user, get_db
from app.modules.staff import service
from app.modules.staff.schemas import (
    StaffListResponse,
    StaffRegisterResponse,
    StaffResponse,
)

router = APIRouter(prefix="/api/staff", tags=["Staff"])

MAX_IMAGES = 5
MAX_IMAGE_BYTES = 10 * 1024 * 1024


@router.post("/register", response_model=StaffRegisterResponse)
async def register_staff(
    images: List[UploadFile] = File(..., description="1-5 clear photos of the staff member's face"),
    name: Optional[str] = Form(None),
    person_identity_id: Optional[UUID] = Form(
        None, description="Mark this existing identity as staff instead of face-searching"
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Register a staff member from photo(s) and set is_staff = TRUE.

    If the face matches an existing identity (face sim >= FACE_MATCH_THRESHOLD) that
    identity is marked staff; otherwise a new staff identity is created.
    """
    if not images or len(images) > MAX_IMAGES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Upload 1-{MAX_IMAGES} images")
    decoded = []
    for img in images:
        data = await img.read()
        if len(data) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{img.filename} exceeds 10 MB"
            )
        decoded.append(service.decode_image(data))
    result = await service.register_staff(
        db, decoded, name.strip() if name else None, person_identity_id
    )
    return StaffRegisterResponse(**result)


@router.get("", response_model=StaffListResponse)
async def list_staff(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all identities with is_staff = TRUE."""
    items = await service.list_staff(db)
    return StaffListResponse(items=[StaffResponse(**i) for i in items], total=len(items))


@router.delete("/{person_identity_id}", status_code=status.HTTP_204_NO_CONTENT)
async def unregister_staff(
    person_identity_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Remove staff status (identity and its history are kept)."""
    await service.unregister_staff(db, person_identity_id)
