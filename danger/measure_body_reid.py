#!/usr/bin/env python3
"""
measure_body_reid.py — Measure OSNet body ReID same/diff distributions.

Downloads body crops from MinIO, extracts FRESH OSNet embeddings with the
currently-loaded model, and computes same-person vs different-person
cosine similarity distributions to validate the ReID model is discriminative.

  Group A — SAME person:  multi-camera persons (cross-camera = same person).
  Group B — DIFFERENT:    concurrent diff-camera pairs (two people in store
                          simultaneously on different cameras = different).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/measure_body_reid.py [--sample 60]
"""

import asyncio
import sys
import argparse
from collections import defaultdict

import numpy as np
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX
from app.modules.reid.osnet_extractor import get_shared_extractor


def cos_sim(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _download_crop(minio_client, crop_path: str):
    if not crop_path:
        return None
    try:
        key = crop_path
        if key.startswith(f"{BUCKET_PREFIX}/"):
            key = key[len(BUCKET_PREFIX) + 1:]
        if "/" in key and not key.startswith("crops/"):
            key = key.split("/", 1)[1]
        resp = minio_client.get_object(BUCKET_PREFIX, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        import cv2
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  ⚠ download failed {crop_path[:60]}: {e}")
        return None


async def measure(sample_limit: int = 60):
    print("\n  Loading OSNet model (fresh weights)...")
    osnet = get_shared_extractor()
    minio_client = get_client()
    print("  OSNet loaded.\n")

    # ── Collect per-person crops, grouped by camera ──────────────────────
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT pi.id::text, ts.camera_id, ts.best_crop_path, ts.total_frames
            FROM person_identities pi
            JOIN track_sessions ts ON ts.person_identity_id = pi.id
            WHERE ts.best_crop_path IS NOT NULL
              AND ts.total_frames >= 5
            ORDER BY pi.id, ts.camera_id, ts.started_at
        """))
        rows = r.fetchall()

    # person_id -> { camera_id -> [crop_paths] }
    person_cam_crops = defaultdict(lambda: defaultdict(list))
    for pid, cam, crop_path, frames in rows:
        person_cam_crops[pid][cam].append(crop_path)

    # ── Group A: multi-camera persons (same person, cross-camera) ────────
    multi_cam_pids = [pid for pid, cams in person_cam_crops.items() if len(cams) >= 2]
    print(f"  Multi-camera persons (same-person ground truth): {len(multi_cam_pids)}")

    # ── Group B: concurrent diff-camera pairs (different persons) ────────
    async with AsyncSessionLocal() as db:
        r = await db.execute(text("""
            SELECT DISTINCT a.person_identity_id::text || '|' || b.person_identity_id::text AS pair
            FROM track_sessions a
            JOIN track_sessions b ON (
              a.person_identity_id < b.person_identity_id
              AND a.camera_id != b.camera_id
              AND a.started_at < b.last_seen_at
              AND b.started_at < a.last_seen_at
              AND a.person_identity_id IS NOT NULL
              AND b.person_identity_id IS NOT NULL
            )
            LIMIT :lim
        """), {"lim": sample_limit})
        concurrent_pairs = [row[0].split("|") for row in r.fetchall()]
    print(f"  Concurrent diff-camera pairs (diff-person ground truth): {len(concurrent_pairs)}\n")

    # ── Extract fresh embeddings: up to 2 crops per camera per person ────
    # Cache: (person_id, camera_id) -> [embeddings]
    emb_cache = {}

    async def get_embeddings(pid, cam, max_crops=2):
        cache_key = (pid, cam)
        if cache_key in emb_cache:
            return emb_cache[cache_key]
        crops = person_cam_crops.get(pid, {}).get(cam, [])[:max_crops]
        embs = []
        for cp in crops:
            img = _download_crop(minio_client, cp)
            if img is not None:
                e = osnet.extract(img)
                if e is not None:
                    embs.append(e)
        emb_cache[cache_key] = embs
        return embs

    # ── Same-person sims: cross-camera within each multi-camera person ────
    same_sims = []
    cam_list_all = list({cam for cams in person_cam_crops.values() for cam in cams})
    print(f"  Cameras in data: {cam_list_all}")

    for pid in multi_cam_pids[:sample_limit]:
        cams = sorted(person_cam_crops[pid].keys())
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                embs_a = await get_embeddings(pid, cams[i])
                embs_b = await get_embeddings(pid, cams[j])
                for ea in embs_a:
                    for eb in embs_b:
                        same_sims.append(cos_sim(ea, eb))

    # ── Diff-person sims: cross-person within concurrent pairs ───────────
    diff_sims = []
    for pid_a, pid_b in concurrent_pairs[:sample_limit]:
        # find a camera for each person (they're on different cameras by definition)
        cams_a = list(person_cam_crops.get(pid_a, {}).keys())
        cams_b = list(person_cam_crops.get(pid_b, {}).keys())
        if not cams_a or not cams_b:
            continue
        embs_a = await get_embeddings(pid_a, cams_a[0])
        embs_b = await get_embeddings(pid_b, cams_b[0])
        for ea in embs_a:
            for eb in embs_b:
                diff_sims.append(cos_sim(ea, eb))

    # ── Report distributions ─────────────────────────────────────────────
    def report(label, sims):
        if not sims:
            print(f"  {label}: no data")
            return
        sims = sorted(sims)
        n = len(sims)
        print(f"  {label}: n={n}  min={sims[0]:.4f}  max={sims[-1]:.4f}  "
              f"p10={sims[n//10]:.4f}  p25={sims[n//4]:.4f}  "
              f"p50={sims[n//2]:.4f}  p75={sims[3*n//4]:.4f}  "
              f"p90={sims[9*n//10]:.4f}  mean={np.mean(sims):.4f}  median={np.median(sims):.4f}")

    print(f"\n{'='*80}")
    print(f"  BODY ReID DISTRIBUTION (fresh OSNet embeddings from MinIO crops)")
    print(f"{'='*80}\n")
    report("SAME-person (cross-camera)", same_sims)
    report("DIFF-person (concurrent)  ", diff_sims)

    # Threshold sweep (F1)
    if same_sims and diff_sims:
        best_f1, best_t = 0, 0
        for t in [x / 100.0 for x in range(0, 100, 1)]:
            tp = sum(1 for s in same_sims if s >= t)
            fn = sum(1 for s in same_sims if s < t)
            fp = sum(1 for s in diff_sims if s >= t)
            tn = sum(1 for s in diff_sims if s < t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        print(f"\n  Best F1 threshold = {best_t:.2f}  (F1={best_f1:.3f})")
        print(f"  Same p10={sorted(same_sims)[len(same_sims)//10]:.4f}  Diff p90={sorted(diff_sims)[9*len(diff_sims)//10]:.4f}")

        # Histogram
        buckets = defaultdict(lambda: {"same": 0, "diff": 0})
        for s in same_sims:
            buckets[int(s * 20) / 20.0]["same"] += 1
        for s in diff_sims:
            buckets[int(s * 20) / 20.0]["diff"] += 1
        print(f"\n  {'Range':<12} {'SAME':>6} {'DIFF':>6}")
        for b in sorted(buckets):
            s, d = buckets[b]["same"], buckets[b]["diff"]
            print(f"  [{b:.2f},{b+0.05:.2f}) {s:>6} {d:>6}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure OSNet body ReID same/diff distributions from fresh crop embeddings.")
    parser.add_argument("--sample", type=int, default=60)
    args = parser.parse_args()
    asyncio.run(measure(args.sample))
