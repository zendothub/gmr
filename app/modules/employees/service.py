"""Employee registration service.

Handles two registration flows:
  Way 1 — Frontend uploads an image:
    • Run InsightFace face detection (quality check)
    • Extract 512-dim ArcFace embedding
    • Cosine-similarity search against all existing PersonFaceEmbeddings
    • If match (≥ FACE_MATCH_THRESHOLD):
        – Update the employee's face_crop_path with the latest image
        – Return already_registered=True + crop URL
    • If no match:
        – Create PersonIdentity + PersonFaceEmbedding
        – Create Employee row
        – Return already_registered=False

  Way 2 — Frontend selects a person from the debug/active-tracks view:
    • Receives person_identity_id (already in DB from camera pipeline)
    • Checks for duplicate emp_id or already-linked person_identity
    • Creates Employee row linked to that person_identity
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List
from uuid import UUID

import cv2
import numpy as np
from fastapi import HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import get_settings
from app.core.db.models.attendance import DEFAULT_WEEKENDS, Employee, Gender, ShiftSlot
from app.core.db.models.person import PersonIdentity, PersonFaceEmbedding
from app.modules.employees.schemas import (
    RegisterByCameraBody,
    EmployeeUpdate,
)
from app.utils.time_utils import utc_now

settings = get_settings()

# Threshold used for "is this person already registered?" check.
# Same as FACE_MATCH_THRESHOLD so it aligns with the live pipeline.
_REGISTRATION_FACE_THRESHOLD = settings.FACE_MATCH_THRESHOLD


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_face_quality(face, det_size_h: int = 640) -> float:
    """Composite face quality: det_score × frontality.

    Frontality is approximated from keypoint eye spread.
    """
    det_score = float(getattr(face, "det_score", 0.0))
    kps = getattr(face, "kps", None)
    if kps is not None and len(kps) >= 2:
        eye_dist = float(np.linalg.norm(np.array(kps[0]) - np.array(kps[1])))
        bbox = face.bbox
        face_w = float(bbox[2] - bbox[0])
        eye_spread = eye_dist / face_w if face_w > 1 else 0.0
        # Normalise to [0,1] — fully frontal ~0.35+; profile ~0.0
        frontality = min(1.0, eye_spread / 0.35)
    else:
        frontality = 0.5  # assume middling frontality

    return det_score * (
        (1.0 - settings.FACE_FRONTALITY_WEIGHT) + settings.FACE_FRONTALITY_WEIGHT * frontality
    )


async def _detect_best_face(image_bytes: bytes):
    """
    Run InsightFace on raw JPEG/PNG bytes.

    Returns (face_obj, face_bgr_crop, embedding_ndarray) or raises HTTPException.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Ensure it is a valid JPEG/PNG.")

    from app.modules.reid.insightface_analyzer import get_shared_analyzer
    analyzer = get_shared_analyzer()
    if analyzer.app is None:
        raise HTTPException(status_code=503, detail="Face recognition model is not available.")

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = analyzer.app.get(rgb)
    if not faces:
        raise HTTPException(status_code=422, detail="No face detected in the uploaded image.")

    # Pick the best face by quality score
    best_face = max(faces, key=lambda f: _compute_face_quality(f))
    quality = _compute_face_quality(best_face)

    if best_face.det_score < settings.FACE_MIN_DET_SCORE:
        raise HTTPException(
            status_code=422,
            detail=f"Face quality too low (det_score={best_face.det_score:.2f}, "
                   f"min={settings.FACE_MIN_DET_SCORE}). Please use a clearer image."
        )

    bbox = best_face.bbox
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    face_crop_bgr = img_bgr[max(0, y1):y2, max(0, x1):x2]

    embedding = getattr(best_face, "embedding", None)
    if embedding is None:
        raise HTTPException(status_code=422, detail="Face detected but embedding extraction failed.")

    embedding = np.array(embedding, dtype=np.float32)
    return best_face, face_crop_bgr, embedding, quality


async def _find_matching_person(
    db: AsyncSession,
    embedding: np.ndarray,
    threshold: float = _REGISTRATION_FACE_THRESHOLD,
) -> Optional[PersonIdentity]:
    """
    Search all PersonFaceEmbeddings for the closest cosine-similarity match.

    Returns the PersonIdentity if any embedding exceeds `threshold`, else None.
    """
    result = await db.execute(
        select(PersonFaceEmbedding).options()
    )
    all_embs: List[PersonFaceEmbedding] = result.scalars().all()

    best_sim = 0.0
    best_person_id: Optional[UUID] = None

    for row in all_embs:
        if row.embedding is None:
            continue
        stored = np.array(row.embedding, dtype=np.float32)
        sim = _cosine_sim(embedding, stored)
        if sim > best_sim:
            best_sim = sim
            best_person_id = row.person_identity_id

    if best_sim >= threshold and best_person_id is not None:
        person = (
            await db.execute(
                select(PersonIdentity).where(PersonIdentity.id == best_person_id)
            )
        ).scalar_one_or_none()
        return person

    return None


async def _upload_face_crop(face_crop_bgr: np.ndarray, prefix: str = "emp_face") -> Optional[str]:
    """Encode and upload a face crop to MinIO; return the object path."""
    try:
        from app.core.db.models.storage import StorageType
        from app.modules.storage import service as storage_svc

        img_bytes = await storage_svc.save_image_bytes(
            face_crop_bgr, StorageType.CROP, prefix=prefix
        )
        if not img_bytes:
            return None

        import uuid as _uuid
        from datetime import datetime as _dt
        ts = _dt.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        obj_name = f"crops/{prefix}_{ts}_{_uuid.uuid4().hex[:8]}.jpg"

        path = await storage_svc.upload_to_storage(img_bytes, StorageType.CROP, obj_name)
        return path
    except Exception as e:
        logger.warning(f"Failed to upload face crop: {e}")
        return None


# ---------------------------------------------------------------------------
# Public service functions
# ---------------------------------------------------------------------------

async def get_employee_by_emp_id(db: AsyncSession, emp_id: str) -> Optional[Employee]:
    return (
        await db.execute(
            select(Employee)
            .where(Employee.emp_id == emp_id)
            .where(Employee.is_active.is_(True))
        )
    ).scalar_one_or_none()


async def register_by_image(
    db: AsyncSession,
    image_bytes: bytes,
    emp_id: str,
    name: str,
    shift_slot_id: Optional[UUID],
    gender: Optional[str] = None,
    weekends: Optional[List[str]] = None,
) -> dict:
    """
    Way 1 — register employee from a directly uploaded image.

    Returns a dict: { employee, already_registered, face_crop_url, message }
    """
    # 1. Detect face & extract embedding
    best_face, face_crop_bgr, embedding, quality = await _detect_best_face(image_bytes)

    # 2. Search DB for existing matching person
    matched_person = await _find_matching_person(db, embedding)

    # 3. Upload the fresh face crop regardless of match (update latest photo)
    crop_path = await _upload_face_crop(face_crop_bgr, prefix="emp_face")

    if matched_person is not None:
        # ── Already registered? ─────────────────────────────────────────────
        # Check whether an ACTIVE Employee row already exists for this person_identity.
        # A soft-deleted (is_active=False) employee is treated as gone — allow re-registration.
        active_employee: Optional[Employee] = (
            await db.execute(
                select(Employee).where(
                    Employee.person_identity_id == matched_person.id,
                    Employee.is_active.is_(True),
                )
            )
        ).scalar_one_or_none()

        if active_employee:
            # Genuinely still active — just refresh the face crop
            if crop_path:
                active_employee.face_crop_path = crop_path
            await db.commit()
            await db.refresh(active_employee)
            await db.execute(
                select(Employee)
                .where(Employee.id == active_employee.id)
                .options(selectinload(Employee.shift_slot))
            )
            return {
                "employee": active_employee,
                "already_registered": True,
                "face_crop_url": crop_path,
                "message": (
                    f"Employee '{active_employee.name}' (emp_id={active_employee.emp_id}) "
                    f"is already registered. Face crop updated."
                ),
            }

        # No active employee linked to this identity.
        # There may be a soft-deleted employee row for the same person — reactivate it
        # rather than creating a duplicate, to preserve attendance history.
        inactive_employee: Optional[Employee] = (
            await db.execute(
                select(Employee).where(
                    Employee.person_identity_id == matched_person.id,
                    Employee.is_active.is_(False),
                )
            )
        ).scalar_one_or_none()

        if inactive_employee:
            # Before reactivating, make sure the new emp_id is not already
            # taken by a different ACTIVE employee.
            if inactive_employee.emp_id != emp_id:
                await _ensure_no_duplicate_emp_id(db, emp_id)
            # Reactivate the old record with the new registration details
            inactive_employee.is_active = True
            inactive_employee.emp_id = emp_id
            inactive_employee.name = name
            if shift_slot_id is not None:
                inactive_employee.shift_slot_id = shift_slot_id
            if gender is not None:
                inactive_employee.gender = Gender(gender)
            if weekends is not None:
                inactive_employee.weekends = weekends
            if crop_path:
                inactive_employee.face_crop_path = crop_path
            await db.commit()
            await db.refresh(inactive_employee)
            await db.execute(
                select(Employee)
                .where(Employee.id == inactive_employee.id)
                .options(selectinload(Employee.shift_slot))
            )
            return {
                "employee": inactive_employee,
                "already_registered": False,
                "face_crop_url": crop_path,
                "message": (
                    f"Face matched an existing identity. Employee '{name}' re-registered "
                    f"successfully (previous record reactivated)."
                ),
            }

        # Person exists in AI DB but has never been an employee — create one now.
        await _ensure_no_duplicate_emp_id(db, emp_id)
        employee = Employee(
            emp_id=emp_id,
            name=name,
            gender=Gender(gender) if gender is not None else None,
            weekends=weekends if weekends is not None else list(DEFAULT_WEEKENDS),
            person_identity_id=matched_person.id,
            shift_slot_id=shift_slot_id,
            face_crop_path=crop_path,
        )
        db.add(employee)
        await db.commit()
        await db.refresh(employee)
        await db.execute(
            select(Employee)
            .where(Employee.id == employee.id)
            .options(selectinload(Employee.shift_slot))
        )
        return {
            "employee": employee,
            "already_registered": False,
            "face_crop_url": crop_path,
            "message": (
                f"Face matched an existing identity. Employee '{name}' registered "
                f"and linked to the existing identity."
            ),
        }
    else:
        # ── New person — create PersonIdentity + embedding + Employee ───────
        await _ensure_no_duplicate_emp_id(db, emp_id)

        # Create PersonIdentity
        person = PersonIdentity(
            label=name,
            is_anonymous=False,
            face_crop_path=crop_path,
            best_face_score=quality,
            first_seen_at=utc_now(),
            last_seen_at=utc_now(),
        )
        db.add(person)
        await db.flush()  # get person.id

        # Store face embedding
        emb_row = PersonFaceEmbedding(
            person_identity_id=person.id,
            embedding=embedding.tolist(),
            face_score=quality,
            face_crop_path=crop_path,
            captured_at=utc_now(),
        )
        db.add(emb_row)

        # Create Employee
        employee = Employee(
            emp_id=emp_id,
            name=name,
            gender=Gender(gender) if gender is not None else None,
            weekends=weekends if weekends is not None else list(DEFAULT_WEEKENDS),
            person_identity_id=person.id,
            shift_slot_id=shift_slot_id,
            face_crop_path=crop_path,
        )
        db.add(employee)

        await db.commit()
        await db.refresh(employee)
        # Eagerly load shift_slot
        await db.execute(
            select(Employee)
            .where(Employee.id == employee.id)
            .options(selectinload(Employee.shift_slot))
        )

        return {
            "employee": employee,
            "already_registered": False,
            "face_crop_url": crop_path,
            "message": f"Employee '{name}' registered successfully.",
        }


async def register_by_camera(
    db: AsyncSession,
    payload: RegisterByCameraBody,
) -> Employee:
    """
    Way 2 — register employee by linking a camera-detected person_identity.

    The frontend picks a person from the debug active-tracks / unique-persons
    view, copies their person_identity_id, and submits it here with emp_id + name.
    """
    # Verify the person_identity exists
    person: Optional[PersonIdentity] = (
        await db.execute(
            select(PersonIdentity).where(PersonIdentity.id == payload.person_identity_id)
        )
    ).scalar_one_or_none()
    if not person:
        raise HTTPException(status_code=404, detail="person_identity not found.")

    # Check person is not already linked to another employee
    already_linked: Optional[Employee] = (
        await db.execute(
            select(Employee).where(
                Employee.person_identity_id == payload.person_identity_id,
                Employee.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()
    if already_linked:
        raise HTTPException(
            status_code=409,
            detail=(
                f"This identity is already linked to employee "
                f"'{already_linked.name}' (emp_id={already_linked.emp_id})."
            ),
        )

    await _ensure_no_duplicate_emp_id(db, payload.emp_id)

    # Use the best face crop from the person_identity as the employee photo
    face_crop_path = person.face_crop_path

    employee = Employee(
        emp_id=payload.emp_id,
        name=payload.name,
        gender=Gender(payload.gender) if payload.gender is not None else None,
        weekends=payload.weekends if payload.weekends is not None else list(DEFAULT_WEEKENDS),
        person_identity_id=payload.person_identity_id,
        shift_slot_id=payload.shift_slot_id,
        face_crop_path=face_crop_path,
    )
    db.add(employee)
    await db.commit()
    await db.refresh(employee)
    # Eagerly load shift_slot
    await db.execute(
        select(Employee)
        .where(Employee.id == employee.id)
        .options(selectinload(Employee.shift_slot))
    )
    return employee


async def list_employees(
    db: AsyncSession,
    page: int = 1,
    size: int = 20,
    search: Optional[str] = None,
    shift_slot_id: Optional[UUID] = None,
    active_only: bool = True,
) -> dict:
    """Paginated employee list with optional filters."""
    from sqlalchemy import or_, func as safunc

    q = select(Employee)
    if active_only:
        q = q.where(Employee.is_active.is_(True))
    if search:
        q = q.where(
            or_(
                Employee.name.ilike(f"%{search}%"),
                Employee.emp_id.ilike(f"%{search}%"),
            )
        )
    if shift_slot_id:
        q = q.where(Employee.shift_slot_id == shift_slot_id)

    total_result = await db.execute(select(safunc.count()).select_from(q.subquery()))
    total = total_result.scalar() or 0

    q = q.options(selectinload(Employee.shift_slot)).order_by(Employee.name).offset((page - 1) * size).limit(size)
    employees = (await db.execute(q)).scalars().all()

    return {"items": employees, "total": total, "page": page, "size": size}


async def get_by_emp_id(db: AsyncSession, emp_id: str) -> Employee:
    emp = (
        await db.execute(
            select(Employee)
            .where(Employee.emp_id == emp_id)
            .options(selectinload(Employee.shift_slot))
        )
    ).scalar_one_or_none()
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee '{emp_id}' not found.")
    return emp


async def update_employee(
    db: AsyncSession, emp_id: str, payload: EmployeeUpdate
) -> Employee:
    emp = await get_by_emp_id(db, emp_id)
    if payload.name is not None:
        emp.name = payload.name
    if payload.gender is not None:
        emp.gender = Gender(payload.gender)
    if payload.weekends is not None:
        emp.weekends = payload.weekends
    if payload.shift_slot_id is not None:
        # Validate shift_slot exists
        slot = (
            await db.execute(
                select(ShiftSlot).where(ShiftSlot.id == payload.shift_slot_id)
            )
        ).scalar_one_or_none()
        if not slot:
            raise HTTPException(status_code=404, detail="Shift slot not found.")
        emp.shift_slot_id = payload.shift_slot_id
    if payload.is_active is not None:
        emp.is_active = payload.is_active
    await db.commit()
    await db.refresh(emp)
    # Eagerly load shift_slot
    await db.execute(
        select(Employee)
        .where(Employee.id == emp.id)
        .options(selectinload(Employee.shift_slot))
    )
    return emp


async def deactivate_employee(db: AsyncSession, emp_id: str) -> None:
    """Soft-delete: set is_active=False. Attendance history is preserved."""
    emp = await get_by_emp_id(db, emp_id)
    emp.is_active = False
    await db.commit()


# ---------------------------------------------------------------------------
# Internal guard
# ---------------------------------------------------------------------------

async def _ensure_no_duplicate_emp_id(db: AsyncSession, emp_id: str) -> None:
    """Raise 409 only when an ACTIVE employee already owns this emp_id.

    Soft-deleted (is_active=False) records are ignored so the same emp_id
    can be reused after an employee has been deactivated.
    """
    existing = (
        await db.execute(
            select(Employee).where(
                Employee.emp_id == emp_id,
                Employee.is_active.is_(True),
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"An employee with emp_id '{emp_id}' already exists.",
        )
