#!/usr/bin/env python3
"""
merge_staff_reattach_duplicates.py — historical backfill for the live staff-reattach path.

Finds non-staff PersonIdentity fragments that should have been reattached to an
is_staff identity by the same rules as IdentityDecisionEngine._try_staff_reattach:

  1. Winner is always is_staff=TRUE; loser is non-staff.
  2. RECENT WINDOW (critical): person or track time ranges are within
     RECENT_WINDOW_MINUTES of each other (default 5). Body-only is only
     trusted inside a same-visit window — clothing constant. Pairs outside
     the window are NEVER merged by this script.
  3. body_median (all cross-pairs frag×staff) ≥ STAFF_REATTACH_BODY_MEDIAN (0.70).
  4. Staff has ≥ STAFF_REATTACH_MIN_BODIES (2) stored body embeddings.
  5. Face required when STAFF_REATTACH_REQUIRE_FACE (default True): fragment must have
     ≥1 face and face_max to staff ≥ STAFF_REATTACH_FACE_MIN (0.30). Faceless rejected.
     Faces that fail cluster fit are still handled by contamination-gated absorb.
  6. Ambiguity: if two staff score within STAFF_REATTACH_AMBIGUITY (0.03) → skip.
  7. No same-camera-overlap hard block (ByteTrack fragments of staff often
     produce brief same-cam dual tracks — live staff reattach omits this gate).

Uses contamination-gated absorb from jobs.tasks. MinIO left to periodic sweep.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_staff_reattach_duplicates.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_staff_reattach_duplicates.py --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_staff_reattach_duplicates.py --ids UUID1 UUID2 --apply
    PYTHONPATH=/gmr/gmr venv/bin/python danger/merge_staff_reattach_duplicates.py --staff-ids UUID --apply
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta

import numpy as np
from loguru import logger
from sqlalchemy import text

from app.config import get_settings
from app.core.db.session import AsyncSessionLocal
from app.modules.jobs.tasks import (
    _absorb_body_embeddings,
    _absorb_face_embeddings,
    _revote_person_gender,
)


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    return np.array(raw, dtype=np.float32)


def _l2normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _median_cross_sim(embs_a: list, embs_b: list) -> float | None:
    if not embs_a or not embs_b:
        return None
    return float(np.median([_cos(a, b) for a in embs_a for b in embs_b]))


def _max_cross_sim(embs_a: list, embs_b: list) -> float | None:
    if not embs_a or not embs_b:
        return None
    return max(_cos(a, b) for a in embs_a for b in embs_b)


def _interval_gap(a_start, a_end, b_start, b_end) -> timedelta:
    """Gap between two closed intervals (0 if they overlap)."""
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return timedelta(days=9999)
    if a_start <= b_end and b_start <= a_end:
        return timedelta(0)
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end


def _windows_within_recent(a_windows, b_windows, max_gap: timedelta) -> bool:
    """True if any track pair is within max_gap (overlap or short gap)."""
    if not a_windows or not b_windows:
        return False
    for a_row in a_windows:
        a_start, a_end = a_row[1], a_row[2]
        for b_row in b_windows:
            b_start, b_end = b_row[1], b_row[2]
            if _interval_gap(a_start, a_end, b_start, b_end) <= max_gap:
                return True
    return False


def _person_span_within_recent(staff: dict, frag: dict, max_gap: timedelta) -> bool:
    """Fallback: person first_seen/last_seen spans within max_gap."""
    return _interval_gap(
        staff["first_seen"], staff["last_seen"],
        frag["first_seen"], frag["last_seen"],
    ) <= max_gap


async def _load_faces(db, pid: str) -> list:
    r = await db.execute(text("""
        SELECT embedding FROM person_face_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": pid})
    out = []
    for row in r.fetchall():
        e = _parse_embedding(row[0])
        if e is not None:
            out.append(_l2normalize(e))
    return out


async def _load_bodies(db, pid: str) -> list:
    r = await db.execute(text("""
        SELECT embedding FROM person_embeddings
        WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
    """), {"pid": pid})
    out = []
    for row in r.fetchall():
        e = _parse_embedding(row[0])
        if e is not None:
            out.append(_l2normalize(e))
    return out


async def _load_tracks(db, pid: str) -> list:
    r = await db.execute(text("""
        SELECT camera_id, started_at, COALESCE(ended_at, last_seen_at) AS ended
        FROM track_sessions
        WHERE person_identity_id::text = :pid
        ORDER BY started_at
    """), {"pid": pid})
    return r.fetchall()


async def run(
    apply_fix: bool,
    ids_filter: list[str] | None,
    staff_ids_filter: list[str] | None,
) -> None:
    settings = get_settings()
    window_min = settings.RECENT_WINDOW_MINUTES
    body_thr = settings.STAFF_REATTACH_BODY_MEDIAN
    min_staff_bodies = settings.STAFF_REATTACH_MIN_BODIES
    face_min = settings.STAFF_REATTACH_FACE_MIN
    require_face = settings.STAFF_REATTACH_REQUIRE_FACE
    ambiguity = settings.STAFF_REATTACH_AMBIGUITY
    max_faces = settings.MAX_FACE_EMBEDDINGS_PER_PERSON
    max_bodies = 10
    max_gap = timedelta(minutes=window_min)

    async with AsyncSessionLocal() as db:
        # Staff
        staff_sql = """
            SELECT id::text, first_seen_at, last_seen_at, best_face_score, visit_count
            FROM person_identities
            WHERE is_staff = TRUE
        """
        staff_params: dict = {}
        if staff_ids_filter:
            staff_sql += " AND id::text = ANY(:sids)"
            staff_params["sids"] = list(staff_ids_filter)
        staff_sql += " ORDER BY first_seen_at"
        r = await db.execute(text(staff_sql), staff_params)
        staff_rows = r.fetchall()

        # Fragments (non-staff)
        frag_sql = """
            SELECT id::text, first_seen_at, last_seen_at, best_face_score, visit_count
            FROM person_identities
            WHERE is_staff = FALSE
        """
        frag_params: dict = {}
        if ids_filter:
            frag_sql += " AND id::text = ANY(:fids)"
            frag_params["fids"] = list(ids_filter)
        frag_sql += " ORDER BY first_seen_at"
        r = await db.execute(text(frag_sql), frag_params)
        frag_rows = r.fetchall()

        print(f"\n{'=' * 88}")
        print(f"  Staff reattach backfill — {'APPLY' if apply_fix else 'DRY RUN'}")
        print(f"  Recent window: {window_min} min | body_median ≥ {body_thr} | "
              f"staff bodies ≥ {min_staff_bodies} | face_min ≥ {face_min} | "
              f"require_face={require_face}")
        print(f"  Staff: {len(staff_rows)}  Fragments (non-staff): {len(frag_rows)}")
        print(f"{'=' * 88}\n")

        if not staff_rows:
            print("  No staff identities. Nothing to do.")
            return

        cache: dict[str, dict] = {}

        async def ensure(pid: str, first, last, score, visits, is_staff: bool):
            if pid in cache:
                return
            faces = await _load_faces(db, pid)
            bodies = await _load_bodies(db, pid)
            tracks = await _load_tracks(db, pid)
            cache[pid] = {
                "first_seen": first,
                "last_seen": last,
                "best_score": float(score or 0.0),
                "visits": visits or 0,
                "faces": faces,
                "bodies": bodies,
                "tracks": tracks,
                "is_staff": is_staff,
            }

        for pid, first, last, score, visits in staff_rows:
            await ensure(pid, first, last, score, visits, True)
        for pid, first, last, score, visits in frag_rows:
            await ensure(pid, first, last, score, visits, False)

        staff_ids = [r[0] for r in staff_rows]
        frag_ids = [r[0] for r in frag_rows]

        # fragment_id → list of (body_median, face_max, staff_id)
        hits: dict[str, list[tuple]] = {}

        for fid in frag_ids:
            fc = cache[fid]
            if not fc["bodies"]:
                continue
            # Faceless fragments never reattach via body-only when require_face
            if require_face and not fc["faces"]:
                continue
            for sid in staff_ids:
                sc = cache[sid]
                if len(sc["bodies"]) < min_staff_bodies:
                    continue
                if require_face and not sc["faces"]:
                    continue

                # ── Recent window gate (required — same-visit only) ────
                # Prefer track-level windows (ByteTrack fragments); fall back
                # to person first/last span if either side has no tracks.
                if fc["tracks"] and sc["tracks"]:
                    if not _windows_within_recent(fc["tracks"], sc["tracks"], max_gap):
                        continue
                else:
                    if not _person_span_within_recent(sc, fc, max_gap):
                        continue

                body_med = _median_cross_sim(fc["bodies"], sc["bodies"])
                if body_med is None or body_med < body_thr:
                    continue

                face_max = (
                    _max_cross_sim(fc["faces"], sc["faces"])
                    if fc["faces"] and sc["faces"]
                    else None
                )
                if require_face:
                    if face_max is None or face_max < face_min:
                        continue
                elif fc["faces"] and sc["faces"]:
                    if face_max is None or face_max < face_min:
                        continue
                elif fc["faces"] and not sc["faces"]:
                    continue

                hits.setdefault(fid, []).append((body_med, face_max, sid))

        merges: list[tuple[str, str, dict]] = []  # (winner_staff, loser_frag, evidence)
        skipped_ambiguous = 0

        for fid, score_list in hits.items():
            score_list.sort(key=lambda x: x[0], reverse=True)
            body_med, face_max, sid = score_list[0]
            if len(score_list) >= 2:
                second = score_list[1][0]
                if (body_med - second) < ambiguity:
                    skipped_ambiguous += 1
                    print(
                        f"  SKIP ambiguous frag={fid[:8]} staff={sid[:8]} "
                        f"body={body_med:.3f} second={second:.3f}"
                    )
                    continue

            drop_face_note = ""
            if face_max is not None and face_max < settings.FACE_CONTAMINATION_THRESHOLD:
                drop_face_note = " (faces will be absorb-gated/dropped if fit < 0.35)"

            ev = {
                "body_median": round(body_med, 4),
                "face_max": round(face_max, 4) if face_max is not None else None,
                "n_frag_faces": len(cache[fid]["faces"]),
                "n_frag_bodies": len(cache[fid]["bodies"]),
                "n_staff_faces": len(cache[sid]["faces"]),
                "n_staff_bodies": len(cache[sid]["bodies"]),
                "note": drop_face_note,
            }
            merges.append((sid, fid, ev))

        print(f"  Accepted merges: {len(merges)}  ambiguous skips: {skipped_ambiguous}\n")
        for staff_id, frag_id, ev in merges:
            print(
                f"  MERGE {frag_id[:8]} → staff {staff_id[:8]}  "
                f"body_med={ev['body_median']} face_max={ev['face_max']} "
                f"frag_f/b={ev['n_frag_faces']}/{ev['n_frag_bodies']} "
                f"staff_f/b={ev['n_staff_faces']}/{ev['n_staff_bodies']}"
                f"{ev['note']}"
            )

        if not apply_fix:
            print(f"\n  Dry run. Pass --apply to execute.")
            print(f"{'=' * 88}\n")
            return

        if not merges:
            print("  Nothing to apply.")
            return

        merged = 0
        for staff_id, frag_id, ev in merges:
            try:
                async with db.begin_nested():
                    for tbl, col in (
                        ("track_sessions", "person_identity_id"),
                        ("events", "person_identity_id"),
                        ("billing_interactions", "person_identity_id"),
                        ("storage_objects", "person_identity_id"),
                    ):
                        await db.execute(
                            text(
                                f"UPDATE {tbl} SET {col} = :winner "
                                f"WHERE {col}::text = :loser"
                            ),
                            {"winner": staff_id, "loser": frag_id},
                        )

                    wc = cache[staff_id]
                    lc = cache[frag_id]
                    update_parts = ["visit_count = visit_count + :extra"]
                    params = {"extra": lc["visits"], "winner": staff_id}
                    if (
                        lc["first_seen"]
                        and wc["first_seen"]
                        and lc["first_seen"] < wc["first_seen"]
                    ):
                        update_parts.append("first_seen_at = :loser_first")
                        params["loser_first"] = lc["first_seen"]
                    if (
                        lc["last_seen"]
                        and wc["last_seen"]
                        and lc["last_seen"] > wc["last_seen"]
                    ):
                        update_parts.append("last_seen_at = :loser_last")
                        params["loser_last"] = lc["last_seen"]
                    await db.execute(
                        text(
                            f"UPDATE person_identities SET {', '.join(update_parts)} "
                            f"WHERE id::text = :winner"
                        ),
                        params,
                    )

                    await _absorb_face_embeddings(db, staff_id, frag_id, max_faces)
                    await _absorb_body_embeddings(db, staff_id, frag_id, max_bodies)
                    await _revote_person_gender(db, staff_id)

                    await db.execute(
                        text("DELETE FROM person_identities WHERE id::text = :loser"),
                        {"loser": frag_id},
                    )

                merged += 1
                print(
                    f"  OK merged {frag_id[:8]} → {staff_id[:8]} "
                    f"(body_med={ev['body_median']})"
                )
            except Exception as e:
                logger.error(f"Staff reattach merge failed {frag_id[:8]} → {staff_id[:8]}: {e}")
                print(f"  FAIL {frag_id[:8]} → {staff_id[:8]}: {e}")

        await db.commit()
        print(f"\n  Applied {merged} merge(s).")
        print(f"{'=' * 88}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Historical staff-reattach merge (recent-window body → is_staff)"
    )
    parser.add_argument("--apply", action="store_true", help="Execute merges")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Limit fragment (non-staff) person UUIDs",
    )
    parser.add_argument(
        "--staff-ids",
        nargs="*",
        default=None,
        help="Limit staff person UUIDs",
    )
    args = parser.parse_args()
    asyncio.run(run(args.apply, args.ids, args.staff_ids))


if __name__ == "__main__":
    main()
