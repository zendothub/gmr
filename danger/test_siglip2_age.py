#!/usr/bin/env python3
"""
test_siglip2_age.py — dry-run SigLIP2 zero-shot age on stored face + body crops.

No DB/config writes. Majority-votes per person across face + body crops.
Prompt set matters a lot (age is NOT what SigLIP2 was trained for) — several
sets are compared.

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_siglip2_age.py
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_siglip2_age.py --set A_coarse5
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from sqlalchemy import text

from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import BUCKET_PREFIX, get_client

PROMPT_SETS: Dict[str, List[Tuple[str, List[str]]]] = {
    "A_coarse5": [
        ("child", ["a photo of a young child", "a small kid", "a child under 12 years old"]),
        ("teenager", ["a photo of a teenager", "an adolescent teen", "a high school age teenager"]),
        ("young_adult", ["a photo of a young adult", "a person in their twenties", "a young man or woman about 25"]),
        ("middle_aged", ["a photo of a middle-aged adult", "a person in their forties or fifties", "a middle-aged man or woman"]),
        ("senior", ["a photo of an elderly senior", "an old person with gray hair", "a senior citizen over 65"]),
    ],
    "B_visual": [
        ("child", ["a child's face", "a kid with a child's face and small body", "a very young child"]),
        ("teenager", ["a teenage face", "a youthful teen face without wrinkles", "a teenager with youthful features"]),
        ("young_adult", ["a young adult face", "smooth face of someone in their twenties", "a young adult customer"]),
        ("middle_aged", ["a middle-aged face with some wrinkles", "an adult face that looks 40 to 55", "a mature adult face"]),
        ("senior", ["an elderly wrinkled face", "an old senior face with gray hair", "a very aged face"]),
    ],
    "C_3way": [
        ("minor_or_teen", ["a child or teenager under 18", "a kid or teen", "a young person under eighteen"]),
        ("adult_working_age", ["an adult between 18 and 60", "a working-age adult", "a grown person who is not elderly"]),
        ("senior", ["an elderly senior over 60", "an old person", "a senior citizen"]),
    ],
    "D_simple": [
        ("child", ["child"]),
        ("teenager", ["teenager"]),
        ("young adult", ["young adult"]),
        ("middle-aged adult", ["middle-aged adult"]),
        ("elderly person", ["elderly person"]),
    ],
    "E_v2_bins": [
        ("under_18", ["a photo of a child", "a photo of a teenager", "a kid or teen under eighteen"]),
        ("age_18_24", ["a young adult about 20 years old", "an 18 to 24 year old person"]),
        ("age_25_34", ["an adult about 30 years old", "a 25 to 34 year old person"]),
        ("age_35_44", ["an adult about 40 years old", "a 35 to 44 year old person"]),
        ("age_45_60", ["an adult about 50 years old", "a 45 to 60 year old person"]),
        ("age_60_plus", ["an elderly senior person", "a person over 60 years old"]),
    ],
}


def minio_key(path: str) -> str:
    if path.startswith(f"{BUCKET_PREFIX}/"):
        return path[len(BUCKET_PREFIX) + 1 :]
    return path


def load_pil(client, path: str) -> Optional[Image.Image]:
    try:
        resp = client.get_object(BUCKET_PREFIX, minio_key(path))
        data = resp.read()
        resp.close()
        resp.release_conn()
    except Exception:
        return None
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        return None
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def majority(votes: Counter, order: List[str]) -> Optional[str]:
    if not votes:
        return None
    top = votes.most_common()
    top_n = top[0][1]
    tied = [k for k, n in top if n == top_n]
    if len(tied) == 1:
        return tied[0]
    tied.sort(key=lambda x: order.index(x) if x in order else 99)
    return tied[0]


class SigLIP2Age:
    def __init__(self, model_id: str = "google/siglip2-base-patch16-224"):
        from transformers import AutoModel, AutoProcessor
        from app.utils.device import get_device

        self.device = get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.logit_scale = (
            float(self.model.logit_scale.exp().item())
            if hasattr(self.model, "logit_scale")
            else 1.0
        )
        self.group_embs: Dict[str, Dict[str, torch.Tensor]] = {}
        print(f"SigLIP2 loaded on {self.device} scale={self.logit_scale:.2f}")

    def prepare(self, set_name: str, groups: List[Tuple[str, List[str]]]) -> None:
        embs = {}
        with torch.no_grad():
            for g, prompts in groups:
                embs[g] = self._encode_texts(prompts)
        self.group_embs[set_name] = embs

    @torch.no_grad()
    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        out = self.model.get_text_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def _encode_image(self, pil: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        out = self.model.get_image_features(**inputs)
        emb = out.pooler_output if hasattr(out, "pooler_output") else out
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    @torch.no_grad()
    def classify(self, set_name: str, pil: Image.Image) -> Tuple[str, Dict[str, float]]:
        img = self._encode_image(pil)
        scores = {}
        for g, temb in self.group_embs[set_name].items():
            scores[g] = float(((img @ temb.T) * self.logit_scale).max().item())
        return max(scores, key=scores.get), scores


async def fetch_persons(db, max_persons: Optional[int]):
    lim = f" LIMIT {int(max_persons)}" if max_persons else ""
    return (
        await db.execute(
            text(
                f"""
                SELECT pi.id::text, pi.estimated_age,
                  (SELECT array_agg(path) FROM (
                     SELECT face_crop_path AS path FROM person_face_embeddings
                     WHERE person_identity_id = pi.id AND face_crop_path IS NOT NULL
                     ORDER BY face_score DESC NULLS LAST LIMIT 4) x) AS faces,
                  (SELECT array_agg(path) FROM (
                     SELECT crop_path AS path FROM person_embeddings
                     WHERE person_identity_id = pi.id AND crop_path IS NOT NULL
                     ORDER BY crop_quality DESC NULLS LAST LIMIT 4) x) AS bodies
                FROM person_identities pi
                ORDER BY pi.created_at NULLS LAST
                {lim}
                """
            )
        )
    ).fetchall()


def print_dist(title: str, c: Counter, order: List[str]) -> None:
    tot = sum(c.values()) or 1
    print(f"  {title} (n={sum(c.values())})")
    for g in order:
        print(f"    {g:20s} {c.get(g, 0):4d} ({100.0 * c.get(g, 0) / tot:5.1f}%)")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="all", help="prompt set name or 'all'")
    ap.add_argument("--max-persons", type=int, default=None)
    args = ap.parse_args()

    sets = (
        PROMPT_SETS
        if args.set == "all"
        else {args.set: PROMPT_SETS[args.set]}
    )

    clf = SigLIP2Age()
    for name, groups in sets.items():
        clf.prepare(name, groups)

    client = get_client()
    async with AsyncSessionLocal() as db:
        people = await fetch_persons(db, args.max_persons)
    print(f"Persons: {len(people)}")

    for set_name, groups in sets.items():
        order = [g for g, _ in groups]
        face_c, body_c, comb_c = Counter(), Counter(), Counter()
        n = 0
        for i, (pid, db_age, faces, bodies) in enumerate(people, 1):
            fv, bv = Counter(), Counter()
            for p in list(faces or []):
                pil = load_pil(client, p)
                if pil is None:
                    continue
                w, _ = clf.classify(set_name, pil)
                fv[w] += 1
            for p in list(bodies or []):
                pil = load_pil(client, p)
                if pil is None:
                    continue
                w, _ = clf.classify(set_name, pil)
                bv[w] += 1
            if not fv and not bv:
                continue
            n += 1
            fm, bm = majority(fv, order), majority(bv, order)
            comb = Counter(); comb.update(fv); comb.update(bv)
            cm = majority(comb, order)
            if fm:
                face_c[fm] += 1
            if bm:
                body_c[bm] += 1
            if cm:
                comb_c[cm] += 1
            if i % 20 == 0:
                print(f"  [{set_name}] {i}/{len(people)}")

        print(f"\n======== {set_name}  persons={n} ========")
        print_dist("FACE majority", face_c, order)
        print_dist("BODY majority", body_c, order)
        print_dist("FACE+BODY majority", comb_c, order)

    print("\nDone. No DB/config changes.")


if __name__ == "__main__":
    asyncio.run(main())
