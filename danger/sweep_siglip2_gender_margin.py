#!/usr/bin/env python3
"""
sweep_siglip2_gender_margin.py — dry-run female-biased decision threshold.

Production decides: best_female_sim vs best_male_sim (whichever max wins).
This sweep keeps production prompts/embeddings and varies:

    margin = male_best - female_best
    predict M only if margin > δ  else F

δ=0 matches production (strict > for F when equal-ish via best_fem > best_mal).
Positive δ → need stronger male evidence (reduces F→M, may add M→F).

GT:
  - hardcoded known F→M IDs → Female
  - others → stored DB gender

No DB / config writes.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/sweep_siglip2_gender_margin.py
"""
from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.reid.siglip2_analyzer import get_shared_siglip2
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

KNOWN_FP_MALE = {
    "3656da5e-cb9a-4215-b898-b44f6db2d59a",
    "3fc4be2c-1996-4e73-838a-ebccf466292f",
    "880d395a-c8af-4f4e-9205-571ce0c0a268",
    "d9eb93f5-92c1-465e-bcf6-9a5fdc44d778",
    "b8565d0b-6021-41ac-87e6-d05da3e0a849",
    "fd027ea1-c52e-44df-83fa-bf86b962ae26",
    "6fc05b89-e359-4176-a986-4369b405eaa0",
    "c08bf35e-2876-4517-bcfc-a3929f30e7ad",
}

# margin = male - female; predict M iff margin > delta
DELTAS = [
    -2.0, -1.0, -0.5, -0.25, -0.1, 0.0,
    0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0,
]


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_bgr(client, path: str) -> Optional[np.ndarray]:
    try:
        r = client.get_object(BUCKET_PREFIX, minio_key(path))
        d = r.read()
        r.close()
        r.release_conn()
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(d, np.uint8), cv2.IMREAD_COLOR)


@torch.no_grad()
def raw_margin(sig, bgr: np.ndarray) -> Optional[float]:
    """Return male_best - female_best using production embeddings."""
    if bgr is None or bgr.size == 0 or sig.model is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    img_emb = sig._encode_image(pil)
    fem = (img_emb @ sig._female_embs.T) * sig._logit_scale
    mal = (img_emb @ sig._male_embs.T) * sig._logit_scale
    bf = float(fem.max().item())
    bm = float(mal.max().item())
    return bm - bf


def decide(margin: float, delta: float) -> str:
    return "M" if margin > delta else "F"


def maj_from_votes(votes: List[str]) -> Optional[str]:
    if not votes:
        return None
    c = Counter(votes)
    m, f = c.get("M", 0), c.get("F", 0)
    if m == f:
        return None
    return "F" if f > m else "M"


async def main() -> None:
    print("Loading SigLIP2 (production prompts)…")
    sig = get_shared_siglip2()
    client = get_client()

    async with AsyncSessionLocal() as db:
        rows = (
            await db.execute(
                text(
                    """
                    SELECT pi.id::text, pi.gender,
                      (SELECT array_agg(path ORDER BY s DESC NULLS LAST) FROM (
                         SELECT face_crop_path AS path, face_score AS s
                         FROM person_face_embeddings
                         WHERE person_identity_id = pi.id
                           AND face_crop_path IS NOT NULL
                         ORDER BY face_score DESC NULLS LAST
                         LIMIT 6
                       ) x) AS faces
                    FROM person_identities pi
                    WHERE pi.gender IS NOT NULL
                    """
                )
            )
        ).fetchall()

    # per person: list of crop margins (male - female)
    people: List[Dict] = []
    for i, (pid, db_g, faces) in enumerate(rows, 1):
        db_g = (db_g or "").strip().upper()[:1]
        if db_g not in ("M", "F"):
            continue
        gt = "F" if pid in KNOWN_FP_MALE else db_g
        margins: List[float] = []
        for p in list(faces or []):
            img = load_bgr(client, p)
            if img is None:
                continue
            m = raw_margin(sig, img)
            if m is not None:
                margins.append(m)
        if not margins:
            continue
        people.append(
            {
                "pid": pid,
                "gt": gt,
                "db": db_g,
                "hard": pid in KNOWN_FP_MALE,
                "margins": margins,
                "mean_m": float(np.mean(margins)),
                "median_m": float(np.median(margins)),
                "best_male_m": float(max(margins)),  # most male-leaning crop
                "best_female_m": float(min(margins)),  # most female-leaning crop
            }
        )
        if i % 20 == 0:
            print(f"  … scored {i}/{len(rows)}")

    print(f"Persons with usable face crops: {len(people)}")
    print(f"  hard F-known: {sum(1 for p in people if p['hard'])}")
    print(f"  gt F: {sum(1 for p in people if p['gt']=='F')}  gt M: {sum(1 for p in people if p['gt']=='M')}")

    # margin distribution
    all_m = [m for p in people for m in p["margins"]]
    mean_by_gt = {
        "F": [p["mean_m"] for p in people if p["gt"] == "F"],
        "M": [p["mean_m"] for p in people if p["gt"] == "M"],
    }
    print("\nMean margin (male−female) by GT person:")
    for g, vals in mean_by_gt.items():
        if not vals:
            continue
        a = np.array(vals)
        print(
            f"  GT={g}: n={len(a)} mean={a.mean():+.2f} median={np.median(a):+.2f} "
            f"p10={np.percentile(a,10):+.2f} p90={np.percentile(a,90):+.2f}"
        )

    # aggregation modes to combine crop margins → person decision
    modes = {
        "mean": lambda p, d: decide(p["mean_m"], d),
        "median": lambda p, d: decide(p["median_m"], d),
        "maj_crops": lambda p, d: maj_from_votes([decide(m, d) for m in p["margins"]]),
        # require most-male crop still only just above delta? use mean (primary)
    }

    def empty():
        return {"n": 0, "c": 0, "w": 0, "a": 0, "f2m": 0, "m2f": 0}

    print("\n" + "=" * 100)
    print("Sweep: predict M iff (male_best − female_best) > δ   else F")
    print("Aggregation over face crops: mean | median | majority of per-crop decisions")
    print("=" * 100)

    best_global = None  # (err_all, m2f, f2m, mode, delta, stats)

    for mode_name, fn in modes.items():
        print(f"\n### aggregation = {mode_name}")
        print(
            f"  {'δ':>6s} {'acc':>7s} {'err':>7s} {'F→M':>5s} {'M→F':>5s} "
            f"{'hard_ok':>8s} {'rest_err':>8s} {'abs':>4s}"
        )
        for delta in DELTAS:
            st_all = empty()
            st_hard = empty()
            st_rest = empty()
            for p in people:
                pred = fn(p, delta)
                for st, use in (
                    (st_all, True),
                    (st_hard, p["hard"]),
                    (st_rest, not p["hard"]),
                ):
                    if not use:
                        continue
                    st["n"] += 1
                    if pred is None:
                        st["a"] += 1
                    elif pred == p["gt"]:
                        st["c"] += 1
                    else:
                        st["w"] += 1
                        if p["gt"] == "F" and pred == "M":
                            st["f2m"] += 1
                        if p["gt"] == "M" and pred == "F":
                            st["m2f"] += 1

            dec = st_all["c"] + st_all["w"]
            acc = 100.0 * st_all["c"] / dec if dec else float("nan")
            err = 100.0 * st_all["w"] / dec if dec else float("nan")
            hard_ok = st_hard["c"]
            hard_n = st_hard["n"]
            rest_dec = st_rest["c"] + st_rest["w"]
            rest_err = 100.0 * st_rest["w"] / rest_dec if rest_dec else float("nan")
            print(
                f"  {delta:+6.2f} {acc:6.1f}% {err:6.1f}% {st_all['f2m']:5d} {st_all['m2f']:5d} "
                f"{hard_ok:3d}/{hard_n:<3d} {rest_err:7.1f}% {st_all['a']:4d}"
            )

            # track best: minimize err, then m2f, then f2m
            if dec and (best_global is None or (err, st_all["m2f"], st_all["f2m"]) < best_global[0]):
                best_global = (
                    (err, st_all["m2f"], st_all["f2m"]),
                    mode_name,
                    delta,
                    st_all,
                    st_hard,
                    st_rest,
                    acc,
                )

    # Focused sweet-spot table for mean aggregation
    print("\n### Recommended candidates (mean crop margin, err sorted)")
    cands = []
    for delta in DELTAS:
        st = empty()
        hard_ok = 0
        for p in people:
            pred = decide(p["mean_m"], delta)
            st["n"] += 1
            if pred == p["gt"]:
                st["c"] += 1
                if p["hard"]:
                    hard_ok += 1
            else:
                st["w"] += 1
                if p["gt"] == "F" and pred == "M":
                    st["f2m"] += 1
                if p["gt"] == "M" and pred == "F":
                    st["m2f"] += 1
        err = 100.0 * st["w"] / st["n"]
        cands.append((err, st["m2f"], st["f2m"], -hard_ok, delta, st, hard_ok))
    cands.sort()
    print(f"  {'δ':>6s} {'err':>7s} {'F→M':>5s} {'M→F':>5s} {'hard_ok':>8s}")
    for err, m2f, f2m, _, delta, st, hard_ok in cands[:12]:
        print(f"  {delta:+6.2f} {err:6.1f}% {f2m:5d} {m2f:5d} {hard_ok:3d}/8")

    if best_global:
        _, mode, delta, st_all, st_hard, st_rest, acc = best_global
        print(
            f"\nBest overall (min err, then M→F, then F→M):\n"
            f"  mode={mode}  δ={delta:+.2f}  acc={acc:.1f}%  "
            f"F→M={st_all['f2m']} M→F={st_all['m2f']}  hard={st_hard['c']}/{st_hard['n']}"
        )

    # detail hard-8 at a few deltas
    print("\n### Hard-8 mean-margin at selected δ")
    show_d = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
    print(f"  {'pid':12s} {'mean_m':>8s}" + "".join(f"  d={d:+.1f}" for d in show_d))
    for p in people:
        if not p["hard"]:
            continue
        line = f"  {p['pid'][:12]:12s} {p['mean_m']:+8.2f}"
        for d in show_d:
            line += f"  {decide(p['mean_m'], d):>5s}"
        print(line)

    # people who flip from M→F as delta rises (true males harmed)
    print("\n### True males that flip F at δ=1.0 / 2.0 (mean)")
    for d in (1.0, 2.0, 3.0):
        harmed = [
            p for p in people
            if p["gt"] == "M" and decide(p["mean_m"], 0.0) == "M" and decide(p["mean_m"], d) == "F"
        ]
        print(f"  δ={d:+.1f}: {len(harmed)} males flipped F")
        for p in harmed[:8]:
            print(f"    {p['pid'][:12]} mean_m={p['mean_m']:+.2f}")

    print("\nDone. No DB/config changes.")


if __name__ == "__main__":
    asyncio.run(main())
