#!/usr/bin/env python3
"""
find_optimal_threshold.py — statistical analysis of cross-person face similarities.

Samples person pairs from two distinct groups to determine the optimal
FACE_MATCH_THRESHOLD for distinguishing same vs. different people on this CCTV:

  Group A — SAME person: identities tracked by BOTH cameras (cross-camera re-ID).
            These ARE the same physical person.  Similarity should be HIGH.
  Group B — DIFFERENT person: identities tracked by DIFFERENT cameras at
            overlapping times.  These are NOT the same person (two people in
            the store simultaneously).  Similarity should be LOW.

Output:
  • Histogram / distribution of similarities for each group
  • Recommended threshold at the intersection of the two distributions
  • Quantiles (p10/p25/p50/p75/p90) for each group

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/find_optimal_threshold.py [--sample 50]
"""

import asyncio
import sys
import argparse
from collections import defaultdict

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal


async def find_threshold(sample_limit: int = 50):
    async with AsyncSessionLocal() as db:
        await db.execute(text("SET LOCAL ivfflat.probes = 50"))

        # ────────────────────────────────────────────────────────────────────
        # Group A — SAME person: identities with tracks on ≥2 cameras
        # ────────────────────────────────────────────────────────────────────
        multi_cam = await db.execute(text("""
            SELECT person_identity_id
            FROM track_sessions
            WHERE person_identity_id IS NOT NULL
            GROUP BY person_identity_id
            HAVING COUNT(DISTINCT camera_id) > 1
            LIMIT :lim
        """), {"lim": sample_limit})
        same_person_ids = [r[0] for r in multi_cam.fetchall()]

        group_a_sims: list[float] = []

        # For each person in Group A, random-sample a DIFFERENT person from
        # Group A and compute face similarity.  Since both are "same person
        # across cameras", we should see HIGH similarity.
        for pid in same_person_ids:
            # Take the best face embedding for this person
            best_face = await db.execute(text("""
                SELECT embedding FROM person_face_embeddings
                WHERE person_identity_id = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC LIMIT 1
            """), {"pid": str(pid)})
            best_row = best_face.fetchone()
            if not best_row:
                continue

            # Find the NEAREST neighbour in ANY other person from Group A
            # (excluding self)
            nn = await db.execute(text("""
                SELECT 1.0 - (a.embedding <=> :emb) AS sim, a.person_identity_id
                FROM person_face_embeddings a
                WHERE a.person_identity_id != :pid
                  AND a.embedding IS NOT NULL
                ORDER BY a.embedding <=> :emb
                LIMIT 3
            """), {"emb": str(best_row[0]), "pid": str(pid)})
            for row in nn.fetchall():
                group_a_sims.append(float(row[0]))

        print(f"\nGroup A — SAME person (multi-camera): {len(same_person_ids)} IDs, {len(group_a_sims)} similarities")

        if group_a_sims:
            group_a_sims.sort()
            n = len(group_a_sims)
            print(f"  p10={group_a_sims[n//10]:.4f}  p25={group_a_sims[n//4]:.4f}  "
                  f"p50={group_a_sims[n//2]:.4f}  p75={group_a_sims[3*n//4]:.4f}  "
                  f"p90={group_a_sims[9*n//10]:.4f}")
            print(f"  min={group_a_sims[0]:.4f}  max={group_a_sims[-1]:.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Group B — DIFFERENT person: concurrent tracks on different cameras
        # ────────────────────────────────────────────────────────────────────
        concurrent = await db.execute(text("""
            SELECT DISTINCT a.person_identity_id AS pid_a, b.person_identity_id AS pid_b
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

        group_b_sims: list[float] = []
        processed = set()

        for pid_a, pid_b in concurrent.fetchall():
            pair_key = (str(pid_a)[:12], str(pid_b)[:12])
            if pair_key in processed:
                continue
            processed.add(pair_key)

            best_a = await db.execute(text("""
                SELECT embedding FROM person_face_embeddings
                WHERE person_identity_id = :pid AND embedding IS NOT NULL
                ORDER BY face_score DESC LIMIT 1
            """), {"pid": str(pid_a)})
            best_a_row = best_a.fetchone()
            if not best_a_row:
                continue

            nn = await db.execute(text("""
                SELECT 1.0 - (a.embedding <=> :emb) AS sim
                FROM person_face_embeddings a
                WHERE a.person_identity_id = :pid_b
                  AND a.embedding IS NOT NULL
                ORDER BY a.embedding <=> :emb
                LIMIT 3
            """), {"emb": str(best_a_row[0]), "pid_b": str(pid_b)})
            for row in nn.fetchall():
                group_b_sims.append(float(row[0]))

        print(f"\nGroup B — DIFFERENT person (concurrent tracks): {len(processed)} pairs, {len(group_b_sims)} similarities")

        if group_b_sims:
            group_b_sims.sort()
            n = len(group_b_sims)
            print(f"  p10={group_b_sims[n//10]:.4f}  p25={group_b_sims[n//4]:.4f}  "
                  f"p50={group_b_sims[n//2]:.4f}  p75={group_b_sims[3*n//4]:.4f}  "
                  f"p90={group_b_sims[9*n//10]:.4f}")
            print(f"  min={group_b_sims[0]:.4f}  max={group_b_sims[-1]:.4f}")

        # ────────────────────────────────────────────────────────────────────
        # Recommended threshold — balance between Group A's lower end and
        # Group B's upper end.  Ideally use p10(A) and p90(B).
        # ────────────────────────────────────────────────────────────────────
        if group_a_sims and group_b_sims:
            a_p10 = group_a_sims[len(group_a_sims)//10]
            b_p90 = group_b_sims[9*len(group_b_sims)//10]

            # Midpoint between "10th percentile of same-person" and
            # "90th percentile of different-person"
            mid = (a_p10 + b_p90) / 2.0

            # Also compute ROC-based optimal: minimize false positives + false negatives
            best_threshold = 0.40  # default
            best_f1 = 0.0

            for t in [x/100.0 for x in range(20, 80, 2)]:
                tp = sum(1 for s in group_a_sims if s >= t)  # same pair, correctly kept
                fn = sum(1 for s in group_a_sims if s < t)   # same pair, incorrectly split
                fp = sum(1 for s in group_b_sims if s >= t)  # diff pair, incorrectly merged
                tn = sum(1 for s in group_b_sims if s < t)   # diff pair, correctly split

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t

            print(f"\n{'='*70}")
            print(f"  Threshold recommendations:")
            print(f"    Midpoint (p10_same + p90_diff) / 2  = {mid:.3f}")
            print(f"    Best F1 (balance FP + FN)           = {best_threshold:.2f}  (F1={best_f1:.3f})")
            print(f"    P10 of SAME person                  = {a_p10:.3f}")
            print(f"    P90 of DIFFERENT person             = {b_p90:.3f}")
            print(f"    Overlap region: [{max(group_a_sims[0], group_b_sims[0]):.3f}, {min(group_a_sims[-1], group_b_sims[-1]):.3f}]")
            print(f"    Distinctness (no overlap): {a_p10 > b_p90}")
            print(f"{'='*70}\n")

            # Print histogram buckets
            buckets = defaultdict(lambda: {"same": 0, "diff": 0})
            for s in group_a_sims:
                bucket = int(s * 20) / 20.0  # 0.05 granularity
                buckets[bucket]["same"] += 1
            for s in group_b_sims:
                bucket = int(s * 20) / 20.0
                buckets[bucket]["diff"] += 1
            print("  Similarity distribution (0.05 buckets):")
            print(f"  {'Range':<10} {'SAME':>8} {'DIFF':>8} {'Ratio':>8}")
            for b in sorted(buckets):
                s = buckets[b]["same"]
                d = buckets[b]["diff"]
                bar = "█" * (s + d)
                ratio = f"{s}/{s+d}" if (s+d) > 0 else "-"
                print(f"  [{b:.2f},{b+0.05:.2f}) {s:>6} {d:>8} {ratio:>8}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find optimal face similarity threshold.")
    parser.add_argument("--sample", type=int, default=50, help="Number of persons/pairs to sample (default 50)")
    args = parser.parse_args()
    asyncio.run(find_threshold(args.sample))
