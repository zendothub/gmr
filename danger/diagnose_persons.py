#!/usr/bin/env python3
"""
diagnose_persons.py — Deep-dive diagnostic for specific person identities.

Answers: Why is the same face appearing across multiple person IDs, and
why do some tracks under these persons contain totally different bodies/faces?

Checks performed:
  1. Cross-person face embedding similarities (are they the same person?)
  2. Intra-person face embedding consistency (contaminated faces?)
  3. Intra-person body embedding clusters (body ReID outliers?)
  4. Track-level metadata (gender, age, duration, timestamps — inconsistencies?)
  5. Track-level identity contradictions (track's face vs stored person faces)
  6. Cross-person body ReID overlap

Usage:
    PYTHONPATH=/gmr/gmr .venv/bin/python danger/diagnose_persons.py \
        b7b7da22-593c-4153-908b-7fb41e284de3 \
        104ef438-5129-4853-913a-a069366f91c2 \
        86be736d-c5a3-4370-b934-a029519eb872 \
        90cd4658-0894-4177-9a59-6d9e0b9dd9d6
"""

import asyncio
import sys
import argparse
import numpy as np
from collections import defaultdict
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.config import get_settings


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _parse_embedding(raw):
    if raw is None:
        return None
    if isinstance(raw, str):
        return np.array(eval(raw), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    # Normalize in case pgvector returns un-normalized vectors
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr


async def diagnose(person_ids: list[str]):
    settings = get_settings()
    ids = list(set(person_ids))
    id_set = set(ids)

    print(f"\n{'='*100}")
    print(f"  PERSON DIAGNOSTIC  ({len(ids)} persons)")
    print(f"{'='*100}")

    async with AsyncSessionLocal() as db:
        # ── 1. Person summary ─────────────────────────────────────────────────
        print("\n\n━━━ PERSON SUMMARY ━━━")
        r = await db.execute(text("""
            SELECT
                pi.id::text,
                pi.first_seen_at,
                pi.last_seen_at,
                pi.visit_count,
                pi.gender,
                pi.estimated_age,
                pi.age_group,
                pi.best_face_score,
                pi.is_staff,
                pi.label,
                (SELECT COUNT(*) FROM person_face_embeddings WHERE person_identity_id = pi.id) AS face_count,
                (SELECT COUNT(*) FROM person_embeddings       WHERE person_identity_id = pi.id) AS body_count,
                (SELECT COUNT(*) FROM track_sessions          WHERE person_identity_id = pi.id) AS track_count,
                (SELECT COUNT(*) FROM track_sessions          WHERE person_identity_id = pi.id AND is_active = TRUE) AS active_tracks
            FROM person_identities pi
            WHERE pi.id::text = ANY(:ids)
            ORDER BY pi.first_seen_at
        """), {"ids": ids})
        person_rows = r.fetchall()
        if not person_rows:
            print("  No persons found for these IDs.")
            return

        col_names = ["ID", "FirstSeen", "LastSeen", "Visits", "Gender",
                     "EstAge", "AgeGroup", "FaceScore", "Staff", "Label",
                     "FaceEmb", "BodyEmb", "Tracks", "Active"]
        fmt = "  {:<38} {:<19} {:<19} {:>6} {:>6} {:>6} {:>10} {:>9} {:>5} {:<8} {:>8} {:>8} {:>6} {:>6}"
        print(fmt.format(*col_names))
        print("  " + "-" * 98)
        person_rows_dict = {}
        for row in person_rows:
            pid = str(row[0])
            person_rows_dict[pid] = row
            print(fmt.format(
                pid[:36],
                str(row[1])[:18] if row[1] else "-",
                str(row[2])[:18] if row[2] else "-",
                row[3] or 0,
                row[4] or "-",
                row[5] or "-",
                row[6] or "-",
                f"{row[7]:.2f}" if row[7] else "-",
                str(row[8]),
                row[9] or "-",
                row[10] or 0,
                row[11] or 0,
                row[12] or 0,
                row[13] or 0,
            ))

        # ── 2. Cross-person face similarities ─────────────────────────────────
        print("\n\n━━━ CROSS-PERSON FACE SIMILARITIES ━━━")
        print("  (Cosine similarity between each pair's face embeddings)")
        await db.execute(text("SET LOCAL ivfflat.probes = 50"))

        all_face_embs = {}
        for pid in ids:
            r = await db.execute(text("""
                SELECT embedding FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC
            """), {"pid": pid})
            embs = [_parse_embedding(row[0]) for row in r.fetchall()]
            if embs:
                all_face_embs[pid] = embs

        if len(all_face_embs) >= 2:
            pids_sorted = sorted(all_face_embs.keys())
            print(f"\n  {'Pair A':<40} {'Pair B':<40} {'Best Pair':>10} {'Min Pair':>10} {'Avg Pair':>10}")
            print("  " + "-" * 98)
            for i in range(len(pids_sorted)):
                for j in range(i + 1, len(pids_sorted)):
                    pa, pb = pids_sorted[i], pids_sorted[j]
                    sims = []
                    for ea in all_face_embs[pa]:
                        for eb in all_face_embs[pb]:
                            sims.append(cos_sim(ea, eb))
                    if sims:
                        best = max(sims)
                        worst = min(sims)
                        avg = sum(sims) / len(sims)
                        flag = " *** SAME PERSON ***" if best >= settings.FACE_MATCH_THRESHOLD else ""
                        print(f"  {pa[:38]:<40} {pb[:38]:<40} {best:>8.4f}   {worst:>8.4f}   {avg:>8.4f}{flag}")
        else:
            print("  Not enough persons with face embeddings to compare.")

        # ── 3. Intra-person face embedding consistency ────────────────────────
        print("\n\n━━━ INTRA-PERSON FACE EMBEDDING CONSISTENCY ━━━")
        print("  (Pairwise similarities within each person's stored face embeddings)")
        print("  Thresholds: MATCH=0.48, CONTAMINATION=0.35, CONTRADICTION=0.25")

        for pid in ids:
            embs = all_face_embs.get(pid, [])
            if len(embs) < 2:
                print(f"\n  {pid[:36]} → {len(embs)} embeddings (skip)")
                continue
            print(f"\n  Person: {pid}")
            N = len(embs)
            contaminated = 0
            high_sim = 0
            low_sim = 0
            for a in range(N):
                sims = [f"{cos_sim(embs[a], embs[b]):.3f}" for b in range(N) if b != a]
                print(f"    face[{a}]: sims_to_others={sims}")
                for b in range(a + 1, N):
                    s = cos_sim(embs[a], embs[b])
                    if s < 0:
                        contaminated += 1
                    elif s < 0.25:
                        low_sim += 1
                    elif s >= 0.48:
                        high_sim += 1
            if contaminated:
                print(f"    ⚠ CONTAMINATED: {contaminated} negative-sim pairs (different person's face)")
            if low_sim:
                print(f"    ⚠ LOW SIMILARITY: {low_sim} pairs below 0.25 (likely different angle or different person)")
            if high_sim == 0 and contaminated == 0 and low_sim == 0:
                print(f"    ✓ All pairs look consistent")

        # ── 4. Intra-person body embedding clusters ───────────────────────────
        print("\n\n━━━ INTRA-PERSON BODY EMBEDDING CLUSTERS ━━━")
        print("  (Pairwise similarities within each person's stored body embeddings)")
        print(f"  Threshold: REID_MATCH_THRESHOLD = {settings.REID_MATCH_THRESHOLD}")

        all_body_embs = {}
        for pid in ids:
            r = await db.execute(text("""
                SELECT embedding, crop_quality FROM person_embeddings
                WHERE person_identity_id::text = :pid AND embedding IS NOT NULL
                ORDER BY crop_quality DESC
            """), {"pid": pid})
            rows = r.fetchall()
            if rows:
                all_body_embs[pid] = [(_parse_embedding(r[0]), r[1]) for r in rows]

        for pid in ids:
            items = all_body_embs.get(pid, [])
            if len(items) < 2:
                print(f"\n  {pid[:36]} → {len(items)} body embeddings (skip)")
                continue
            print(f"\n  Person: {pid} ({len(items)} body embeddings)")
            N = len(items)
            sims_matrix = np.zeros((N, N), dtype=np.float32)
            for a in range(N):
                for b in range(N):
                    sims_matrix[a][b] = cos_sim(items[a][0], items[b][0])

            # Print top 20 closest and farthest pairs
            pairs = []
            for a in range(N):
                for b in range(a + 1, N):
                    pairs.append((a, b, sims_matrix[a][b]))
            pairs.sort(key=lambda x: x[2])

            print(f"    Self-sim range: {pairs[0][2]:.4f} to {pairs[-1][2]:.4f}")
            print(f"    avg={np.mean([p[2] for p in pairs]):.4f}, median={np.median([p[2] for p in pairs]):.4f}, std={np.std([p[2] for p in pairs]):.4f}")

            # Flag pairs below threshold
            below_thresh = [(a, b, s) for a, b, s in pairs if s < 0.40]
            very_low = [(a, b, s) for a, b, s in pairs if s < 0.10]
            if very_low:
                print(f"    ⚠ VERY LOW SIMILARITY (<0.10): {len(very_low)} pair(s) — LIKELY DIFFERENT PERSON")
                for a, b, s in very_low[:5]:
                    print(f"      emb[{a}](qual={items[a][1]:.2f}) ↔ emb[{b}](qual={items[b][1]:.2f}) = {s:.4f}")
            elif below_thresh:
                print(f"    ⚠ LOW SIMILARITY (<0.40): {len(below_thresh)} pair(s)")
                for a, b, s in below_thresh[:5]:
                    print(f"      emb[{a}](qual={items[a][1]:.2f}) ↔ emb[{b}](qual={items[b][1]:.2f}) = {s:.4f}")

            # Show all similarities concisely
            if N <= 15:
                for a in range(N):
                    sims = [f"{sims_matrix[a][b]:.3f}" for b in range(N) if b != a]
                    print(f"    body[{a}](qual={items[a][1]:.2f}): sims_to_others={sims}")

        # ── 5. Track-level metadata & inconsistencies ─────────────────────────
        print("\n\n━━━ TRACK-LEVEL DETAIL ━━━")
        print("  (Gender, age, timestamps, and durations per track)")

        for pid in ids:
            r = await db.execute(text("""
                SELECT
                    ts.id::text,
                    ts.camera_id::text,
                    ts.local_track_id,
                    ts.gender,
                    ts.age_group,
                    ts.started_at,
                    ts.last_seen_at,
                    ts.ended_at,
                    ts.total_frames,
                    ts.avg_confidence,
                    ts.stability_score,
                    ts.is_active,
                    ts.best_crop_path
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                ORDER BY ts.started_at
            """), {"pid": pid})
            tracks = r.fetchall()
            print(f"\n  Person: {pid} ({len(tracks)} tracks)")

            if not tracks:
                print("    (no tracks)")
                continue

            # Gender distribution
            genders = [t[3] for t in tracks if t[3] is not None]
            if genders:
                from collections import Counter
                gc = Counter(genders)
                print(f"    Gender distribution: {dict(gc)}")
                if len(gc) > 1:
                    print(f"    ⚠ INCONSISTENT GENDER — tracks of same person have different genders!")

            for t in tracks:
                tid = str(t[0])[:12]
                cam = str(t[1])[:12] if t[1] else "-"
                gender = t[3] or "-"
                age = t[4] or "-"
                started = str(t[5])[:19] if t[5] else "-"
                last = str(t[6])[:19] if t[6] else "-"
                ended = str(t[7])[:19] if t[7] else "-"
                frames = t[8] or 0
                conf = f"{t[9]:.2f}" if t[9] else "-"
                stability = f"{t[10]:.2f}" if t[10] else "-"
                active = "ACTIVE" if t[11] else "ended"
                crop = str(t[12])[:40] if t[12] else "-"

                # Compute duration in seconds from started_at to last_seen_at
                duration = ""
                if t[5] and t[6]:
                    secs = (t[6] - t[5]).total_seconds()
                    duration = f"{secs:.0f}s"

                print(f"    trk={tid} cam={cam} {active:6} g={gender} age={age} started={started} last={last} dur={duration} frames={frames} conf={conf} stab={stability}")
                if t[12]:
                    print(f"          crop={crop}")

        # ── 6. Track identity contradiction check ─────────────────────────────
        print("\n\n━━━ IDENTITY CONTRADICTION CHECK ━━━")
        print(f"  (For each person: do tracks' best crop faces contradict stored person faces?)")
        print(f"  CONTRADICTION threshold = {settings.FACE_CONTRADICTION_THRESHOLD}")

        for pid in ids:
            stored_faces = all_face_embs.get(pid, [])
            if not stored_faces:
                print(f"\n  {pid[:36]} → no stored face embeddings, skip")
                continue

            # Find tracks with best_crop_path (might have face data)
            r = await db.execute(text("""
                SELECT ts.id::text, ts.best_crop_path
                FROM track_sessions ts
                WHERE ts.person_identity_id::text = :pid
                  AND ts.best_crop_path IS NOT NULL
            """), {"pid": pid})
            crop_tracks = r.fetchall()

            if not crop_tracks:
                print(f"\n  {pid[:36]} → {len(stored_faces)} faces, no track crops")
                continue

            print(f"\n  Person: {pid[:36]} ({len(stored_faces)} faces, {len(crop_tracks)} tracks with crops)")

        # ── 7. Cross-person body ReID overlap ─────────────────────────────────
        print("\n\n━━━ CROSS-PERSON BODY REID OVERLAP ━━━")
        print(f"  (Cosine similarity between body embeddings across different persons)")
        print(f"  REID_MATCH_THRESHOLD = {settings.REID_MATCH_THRESHOLD}")

        if len(all_body_embs) >= 2:
            pids_sorted = sorted(all_body_embs.keys())
            print(f"\n  {'Pair A':<40} {'Pair B':<40} {'Best':>10} {'Min':>10} {'Avg':>10} {'N_pairs':>8}")
            print("  " + "-" * 98)
            issues = []
            for i in range(len(pids_sorted)):
                for j in range(i + 1, len(pids_sorted)):
                    pa, pb = pids_sorted[i], pids_sorted[j]
                    sims = []
                    for ea, _ in all_body_embs[pa]:
                        for eb, _ in all_body_embs[pb]:
                            sims.append(cos_sim(ea, eb))
                    if sims:
                        best = max(sims)
                        worst = min(sims)
                        avg = sum(sims) / len(sims)
                        flag = ""
                        if best >= settings.REID_MATCH_THRESHOLD:
                            flag = " *** CROSS-PERSON MATCH (should be different people) ***"
                            issues.append((pa, pb, best))
                        print(f"  {pa[:38]:<40} {pb[:38]:<40} {best:>8.4f}   {worst:>8.4f}   {avg:>8.4f}   {len(sims):>6}{flag}")
            if issues:
                print(f"\n  ⚠ {len(issues)} cross-person pairs exceed REID_MATCH_THRESHOLD — body ReID can't separate them!")
        else:
            print("  Not enough persons with body embeddings to compare.")

    print(f"\n{'='*100}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep-dive diagnostic for person identities")
    parser.add_argument("ids", nargs="+", help="Person identity UUIDs to diagnose")
    args = parser.parse_args()
    asyncio.run(diagnose(args.ids))
