"""Staff registration service.

Replaces the automatic attendance-based staff classifier
(app/modules/reid/staff_classifier.py — consec-5 / 11-of-15 days, now disabled).

Flow:
  1. Detect the face in each uploaded registration photo (InsightFace SCRFD + ArcFace).
  2. Search ALL stored face embeddings for an existing identity (face sim >= FACE_MATCH_THRESHOLD).
     - match  → mark that identity is_staff=TRUE (keeps its tracking history)
     - none   → create a new identity with is_staff=TRUE
  3. Store the registration face embeddings on that identity.

After this, the live pipeline is unchanged: decide_identity's face search is global
(no time window), so the staff member's CCTV tracks match the registered identity
and inherit is_staff — excluded from footfall/purchase analytics, eligible for staff reattach.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import List, Optional, Tuple

import cv2
import numpy as np
from fastapi import HTTPException, status
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.db.models.person import PersonFaceEmbedding, PersonIdentity
from app.modules.reid.identity_decision_engine import IDENTITY_ADVISORY_LOCK_KEY
from app.utils.time_utils import utc_now

# Registration photos come from a phone/webcam, not CCTV — reject a photo if a second
# face is at least this fraction of the main face's area (can't tell who is staff).
SECOND_FACE_AREA_RATIO = 0.5
# Registration faces are pinned (never pruned by the CCTV face cap / cleanup); keep the
# best N across repeated registrations so they can't grow unbounded.
MAX_REGISTRATION_FACES = 5

# Photo quality gate for registration (POST /api/staff/check-image + register):
# blur only (size / distance / pose / lighting are reported in metrics, not enforced).
# Laplacian variance of the 112x112 grey face; first real photos: blurry=39, clear=100-370.
MIN_BLUR_SCORE = 50.0


def _normalize(emb: np.ndarray) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float32)
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def decode_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Could not decode image")
    return img


def _frontality(kps, bbox: dict) -> float:
    """0 = profile, 1 = frontal. Same formula as InsightFaceAnalyzer.analyze()."""
    if kps is None or len(kps) < 2:
        return 0.5  # unknown
    face_w = max(bbox["x2"] - bbox["x1"], 1.0)
    face_h = max(bbox["y2"] - bbox["y1"], 1.0)
    eye_spread = abs(float(kps[1][0]) - float(kps[0][0])) / face_w
    spread_score = min(1.0, eye_spread / 0.35)
    if len(kps) < 5:
        return spread_score
    face_cx = (bbox["x1"] + bbox["x2"]) / 2.0
    nose_score = max(0.0, 1.0 - abs(float(kps[2][0]) - face_cx) / (face_w / 2.0))
    sym_score = max(0.0, 1.0 - abs(float(kps[1][1]) - float(kps[0][1])) / face_h * 4.0)
    return 0.55 * spread_score + 0.30 * nose_score + 0.15 * sym_score


def assess_image(img: np.ndarray) -> dict:
    """Quality report for one registration photo. Runs in a worker thread.

    Every check is pass/fail; any "fail" blocks registration. Returns
    {"ok", "face_count", "metrics", "checks", "face"} — "face" is the internal
    extracted face (embedding/crop) when ok, else None.
    """
    from app.modules.reid.insightface_analyzer import get_shared_analyzer

    settings = get_settings()
    checks: List[dict] = []
    metrics: dict = {}

    def check(name: str, state: str, message: str) -> None:
        checks.append({"name": name, "status": state, "message": message})

    faces = get_shared_analyzer().detect_all_faces(img)
    faces = [f for f in faces if f.get("embedding") is not None]
    result = {"ok": False, "face_count": len(faces), "metrics": metrics, "checks": checks, "face": None}
    if not faces:
        check("face_detected", "fail",
              "No face detected — the whole face must be visible, well lit and facing the camera")
        return result

    def _area(f):
        b = f["bbox"]
        return (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])

    faces.sort(key=_area, reverse=True)
    main = faces[0]
    b = main["bbox"]
    h, w = img.shape[:2]
    face_w = int(b["x2"] - b["x1"])
    pad_x = (b["x2"] - b["x1"]) * 0.30
    pad_y = (b["y2"] - b["y1"]) * 0.30
    x1, y1 = max(0, int(b["x1"] - pad_x)), max(0, int(b["y1"] - pad_y))
    x2, y2 = min(w, int(b["x2"] + pad_x)), min(h, int(b["y2"] + pad_y))
    crop = img[y1:y2, x1:x2]
    # Blur/brightness on the face only, resized so the score doesn't depend on photo size.
    gray = cv2.cvtColor(
        cv2.resize(img[int(b["y1"]):int(b["y2"]), int(b["x1"]):int(b["x2"])], (112, 112)),
        cv2.COLOR_BGR2GRAY,
    )
    metrics.update(
        {
            "det_score": round(float(main["det_score"]), 3),
            "face_width_px": face_w,
            "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
            "frontality": round(_frontality(main.get("kps"), b), 3),
            "brightness": round(float(gray.mean()), 1),
        }
    )

    if len(faces) > 1 and _area(faces[1]) >= SECOND_FACE_AREA_RATIO * _area(main):
        check("single_face", "fail", f"{len(faces)} faces detected — use a photo with only the staff member")
    else:
        check("single_face", "pass", "One face")

    if metrics["det_score"] < settings.FACE_IDENTITY_MIN_SCORE:
        check("detection_confidence", "fail",
              f"Face detection confidence too low ({metrics['det_score']:.2f} < {settings.FACE_IDENTITY_MIN_SCORE})")
    else:
        check("detection_confidence", "pass", f"Confidence {metrics['det_score']:.2f}")

    if metrics["blur_score"] < MIN_BLUR_SCORE:
        check("sharpness", "fail", "Photo is blurry — hold the camera steady and retake")
    else:
        check("sharpness", "pass", f"Sharpness {metrics['blur_score']}")

    result["ok"] = not any(c["status"] == "fail" for c in checks)
    if result["ok"]:
        result["face"] = {
            "embedding": _normalize(main["embedding"]),
            "det_score": float(main["det_score"]),
            "age": main.get("age"),
            "crop": crop.copy(),
        }
    return result


def _extract_registration_face(img: np.ndarray) -> Tuple[Optional[dict], Optional[str]]:
    """Return (face_dict, None) or (None, reason). Runs in a worker thread."""
    report = assess_image(img)
    if report["face"] is None:
        return None, "; ".join(c["message"] for c in report["checks"] if c["status"] == "fail")
    return report["face"], None


async def extract_faces(images: List[np.ndarray]) -> Tuple[List[dict], List[str]]:
    """Extract one face per image; all photos must pass quality and show one person.

    Raises 422 listing every problem if any photo is rejected — nothing is stored.
    """
    settings = get_settings()
    faces: List[dict] = []
    rejected: List[str] = []
    for i, img in enumerate(images):
        face, reason = await asyncio.to_thread(_extract_registration_face, img)
        if face is None:
            rejected.append(f"image {i + 1}: {reason}")
        else:
            face["index"] = i
            faces.append(face)

    # Multiple photos must be the same person: drop faces whose median sim to the
    # others is below the contamination threshold (same rule as live face storage).
    if len(faces) >= 3:
        embs = np.stack([f["embedding"] for f in faces])
        sims = embs @ embs.T
        keep = []
        for j, f in enumerate(faces):
            others = np.delete(sims[j], j)
            med = float(np.median(others))
            if med >= settings.FACE_CONTAMINATION_THRESHOLD:
                keep.append(f)
            else:
                rejected.append(
                    f"image {f['index'] + 1}: face does not match the other photos "
                    f"(median sim {med:.2f})"
                )
        faces = keep
    elif len(faces) == 2:
        sim = float(faces[0]["embedding"] @ faces[1]["embedding"])
        if sim < settings.FACE_CONTAMINATION_THRESHOLD:
            rejected.append(f"the two photos look like different people (face sim {sim:.2f})")
    if rejected:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, _reject_message(rejected))
    return faces, rejected


def _reject_message(reasons: List[str]) -> str:
    return "Photo not accepted: " + "; ".join(r[0].upper() + r[1:] for r in reasons)


async def _find_matching_person(
    db: AsyncSession, faces: List[dict]
) -> Tuple[Optional[uuid.UUID], Optional[float]]:
    """Best existing identity across all registration faces (global, no time window)."""
    settings = get_settings()
    await db.execute(text("SET LOCAL ivfflat.probes = 50"))
    best_pid, best_sim = None, None
    for f in faces:
        row = (
            await db.execute(
                text(
                    """
                    SELECT person_identity_id, 1 - (embedding <=> :emb) AS sim
                    FROM person_face_embeddings
                    ORDER BY embedding <=> :emb
                    LIMIT 1
                    """
                ),
                {"emb": str(f["embedding"].tolist())},
            )
        ).first()
        if row is not None and (best_sim is None or float(row[1]) > best_sim):
            best_pid, best_sim = row[0], float(row[1])
    if best_sim is not None and best_sim >= settings.FACE_MATCH_THRESHOLD:
        return best_pid, best_sim
    return None, best_sim


async def check_images(db: AsyncSession, images: List[np.ndarray]) -> dict:
    """Pre-registration quality check. Read-only: stores nothing."""
    settings = get_settings()
    reports = []
    for i, img in enumerate(images):
        report = await asyncio.to_thread(assess_image, img)
        report["index"] = i + 1
        reports.append(report)

    faces = [r["face"] for r in reports if r["face"] is not None]
    same_person: Optional[bool] = None
    if len(faces) >= 2:
        embs = np.stack([f["embedding"] for f in faces])
        sims = embs @ embs.T
        same_person = bool(sims[np.triu_indices(len(faces), 1)].min() >= settings.FACE_CONTAMINATION_THRESHOLD)

    existing_match = None
    if faces:
        pid, sim = await _find_matching_person(db, faces)
        if pid is not None:
            person = await db.get(PersonIdentity, pid)
            existing_match = {
                "person_identity_id": pid,
                "similarity": round(sim, 3),
                "name": person.label if person else None,
                "is_staff": bool(person.is_staff) if person else False,
            }
        await db.rollback()  # release SET LOCAL / read transaction

    for r in reports:
        r.pop("face", None)
    return {
        "ok": bool(faces) and all(r["ok"] for r in reports) and same_person is not False,
        "images": reports,
        "same_person": same_person,
        "existing_match": existing_match,
    }


async def _upload_crop(person_id: uuid.UUID, crop: np.ndarray) -> Optional[str]:
    from app.modules.storage.minio_client import upload_image

    # Outside the crops/ prefix so the orphan sweep never touches registration photos.
    object_name = f"staff/{person_id}/{uuid.uuid4().hex}.jpg"
    return await asyncio.to_thread(upload_image, crop, object_name)


async def register_staff(
    db: AsyncSession,
    images: List[np.ndarray],
    name: Optional[str],
    person_identity_id: Optional[uuid.UUID] = None,
) -> dict:
    faces, rejected = await extract_faces(images)  # raises 422 on any bad photo

    now = utc_now()
    # Same lock as live decide_identity / faceless delete — no race with identity create/delete.
    await db.execute(text(f"SELECT pg_advisory_xact_lock({IDENTITY_ADVISORY_LOCK_KEY})"))

    match_sim: Optional[float] = None
    if person_identity_id is not None:
        person = await db.get(PersonIdentity, person_identity_id)
        if person is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Person identity not found")
        action = "linked"
    else:
        pid, match_sim = await _find_matching_person(db, faces)
        if pid is not None:
            person = await db.get(PersonIdentity, pid)
            action = "linked"
        else:
            person = None
            action = "created"

    # Same-angle duplicate guard (as live face storage): skip a face > 0.95 sim to one
    # already stored on this identity, e.g. the same photo registered twice.
    existing = []
    if person is not None:
        existing = [
            _normalize(np.array(json.loads(r[0]) if isinstance(r[0], str) else r[0], dtype=np.float32))
            for r in (
                await db.execute(
                    text("SELECT embedding FROM person_face_embeddings WHERE person_identity_id = :pid"),
                    {"pid": person.id},
                )
            ).fetchall()
        ]
    new_faces = []
    for f in faces:
        if any(float(f["embedding"] @ e) > 0.95 for e in existing):
            rejected.append(f"image {f['index'] + 1}: duplicate of an already stored face")
            continue
        existing.append(f["embedding"])
        new_faces.append(f)
    # Nothing new and already staff → nothing to do. (Not yet staff, e.g. after
    # DELETE /api/staff/{id}: re-registering with the same photo re-marks them.)
    if not new_faces and person is not None and person.is_staff:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"Already registered as staff ({person.label or str(person.id)[:8]}) with "
            f"{'this photo' if len(faces) == 1 else 'these photos'} — upload a different photo",
        )

    best = max(faces, key=lambda f: f["det_score"])
    if person is None:
        person = PersonIdentity(
            label=name,
            first_seen_at=now,
            last_seen_at=now,
            visit_count=0,
            is_anonymous=False,
            is_staff=True,
            best_face_score=best["det_score"],
            estimated_age=best.get("age"),
        )
        db.add(person)
        await db.flush()

    person.is_staff = True
    person.is_anonymous = False
    if name:
        person.label = name
    meta = dict(person.metadata_json or {})
    meta.update(
        {
            "staff_registered": True,
            "staff_registered_at": now.isoformat(),
            "staff_name": name or meta.get("staff_name"),
        }
    )
    person.metadata_json = meta

    stored = 0
    for f in new_faces:
        crop_path = await _upload_crop(person.id, f["crop"])
        db.add(
            PersonFaceEmbedding(
                person_identity_id=person.id,
                embedding=f["embedding"].tolist(),
                camera_id=None,
                face_score=f["det_score"],
                face_crop_path=crop_path,
                captured_at=now,
                is_registration=True,
            )
        )
        stored += 1
        if f is best and crop_path and (
            person.face_crop_path is None
            or (person.best_face_score or 0.0) <= f["det_score"]
        ):
            person.face_crop_path = crop_path
            person.best_face_score = f["det_score"]
    await db.flush()

    # Registration faces are pinned outside MAX_FACE_EMBEDDINGS_PER_PERSON (CCTV faces
    # keep their own cap); only bound the registration set itself.
    await db.execute(
        text(
            """
            DELETE FROM person_face_embeddings WHERE id IN (
                SELECT id FROM person_face_embeddings
                WHERE person_identity_id = :pid AND is_registration
                ORDER BY face_score DESC, captured_at DESC
                OFFSET :keep
            )
            """
        ),
        {"pid": person.id, "keep": MAX_REGISTRATION_FACES},
    )
    await db.commit()

    logger.info(
        f"Staff registered: person={str(person.id)[:8]} action={action} name={name!r} "
        f"faces={stored} rejected={len(rejected)} match_sim="
        f"{f'{match_sim:.3f}' if match_sim is not None else 'n/a'}"
    )
    return {
        "person_identity_id": person.id,
        "name": person.label,
        "action": action,
        "match_similarity": match_sim,
        "faces_stored": stored,
        "faces_rejected": len(rejected),
        "rejected": rejected,
    }


async def list_staff(db: AsyncSession) -> List[dict]:
    rows = (
        await db.execute(
            text(
                """
                SELECT pi.id, pi.label, pi.is_staff,
                       COALESCE((pi.metadata_json->>'staff_registered')::boolean, FALSE),
                       (SELECT COUNT(*) FROM person_face_embeddings fe
                        WHERE fe.person_identity_id = pi.id),
                       (SELECT COUNT(*) FROM person_face_embeddings fe
                        WHERE fe.person_identity_id = pi.id AND fe.is_registration),
                       pi.first_seen_at, pi.last_seen_at, pi.face_crop_path
                FROM person_identities pi
                WHERE pi.is_staff = TRUE
                ORDER BY pi.label NULLS LAST, pi.first_seen_at
                """
            )
        )
    ).fetchall()
    return [
        {
            "person_identity_id": r[0],
            "name": r[1],
            "is_staff": r[2],
            "registered": r[3],
            "face_count": int(r[4]),
            "registration_face_count": int(r[5]),
            "first_seen_at": r[6],
            "last_seen_at": r[7],
            "face_crop_path": r[8],
        }
        for r in rows
    ]


async def unregister_staff(db: AsyncSession, person_identity_id: uuid.UUID) -> None:
    """Demote to customer. Identity, faces and history are kept."""
    person = await db.get(PersonIdentity, person_identity_id)
    if person is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Person identity not found")
    person.is_staff = False
    meta = dict(person.metadata_json or {})
    meta["staff_registered"] = False
    meta["staff_unregistered_at"] = utc_now().isoformat()
    person.metadata_json = meta
    await db.commit()
    logger.info(f"Staff unregistered: person={str(person_identity_id)[:8]}")
