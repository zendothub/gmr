#!/usr/bin/env python3
"""test_siglip2_body.py — SigLIP2 body crop similarity on known identity groups.

Tests whether SigLIP2's image embeddings can separate SAME person from
DIFFERENT person body crops better than OSNet (which produces overlapping
0.58-0.83 similarity ranges for both).

Usage:
    PYTHONPATH=/gmr/gmr venv/bin/python danger/test_siglip2_body.py [--ids id1 id2 ...]
"""

import asyncio, cv2, numpy as np, sys, io, argparse
from PIL import Image
from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX

DEFAULT_IDS = [
    "46242f12-ebf3-4c16-9f08-48ab00bf61b0",  # 3 mixed people, user confirmed
    "c24d1dd0-081d-47f8-91f4-e2c1b439f2f1",  # contaminated, staff
    "a2e1c283-2e08-40c9-8519-49219055797a",  # has face + body
    "bafc6048-e94e-4d94-9eb2-f32d19d4b7f8",  # has face + body
]

async def run(ids: list[str]):
    from transformers import AutoProcessor, AutoModel
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/siglip2-base-patch16-224"

    print(f"Loading SigLIP2 on {device}...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    print("Loaded.\n")

    client = get_client()

    async with AsyncSessionLocal() as db:
        for pid in ids:
            # Get body crops + face crops
            r = await db.execute(text("""
                SELECT crop_path, crop_quality, captured_at
                FROM person_embeddings
                WHERE person_identity_id::text = :pid AND crop_path IS NOT NULL
                ORDER BY captured_at
            """), {"pid": pid})
            body_rows = r.fetchall()
            r = await db.execute(text("""
                SELECT face_crop_path, face_score, captured_at
                FROM person_face_embeddings
                WHERE person_identity_id::text = :pid AND face_crop_path IS NOT NULL
                ORDER BY captured_at
            """), {"pid": pid})
            face_rows = r.fetchall()

            r = await db.execute(text("SELECT gender, visit_count FROM person_identities WHERE id::text=:pid"), {"pid": pid})
            meta = r.fetchone()
            if not meta:
                print(f"  {pid[:12]}: NOT FOUND\n")
                continue

            print(f"\n{'='*65}")
            print(f"  {pid[:12]}  DB_gender={meta[0]}  visits={meta[1]}")
            print(f"  Body crops: {len(body_rows)}  Face crops: {len(face_rows)}")
            print(f"{'='*65}")

            # Load body crops
            body_embs = []
            body_labels = []
            for path, quality, captured in body_rows:
                key = path.split("/", 1)[1] if "/" in path else path
                key = key[len(BUCKET_PREFIX)+1:] if key.startswith(f"{BUCKET_PREFIX}/") else key
                try:
                    resp = client.get_object(BUCKET_PREFIX, key)
                    data = resp.read(); resp.close(); resp.release_conn()
                    img = Image.open(io.BytesIO(data)).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        emb = model.get_image_features(**inputs)
                        emb = emb / emb.norm(dim=-1, keepdim=True)
                    body_embs.append(emb.cpu().numpy())
                    body_labels.append(f"B{captured.strftime('%H:%M')} q={quality:.2f}")
                except Exception as e:
                    body_embs.append(None)
                    body_labels.append(f"B{captured.strftime('%H:%M')} err")

            # Compute pairwise body-body similarity
            N = len(body_embs)
            if N >= 2:
                print(f"\n  Body-body SigLIP2 cosine similarity matrix:")
                header = "       " + "".join(f"{i:>7d}" for i in range(N))
                print(f"  {header}")
                for i in range(N):
                    row = f"  [{i}]  "
                    for j in range(N):
                        if i == j:
                            row += "   1.00"
                        elif body_embs[i] is None or body_embs[j] is None:
                            row += "   ERR "
                        else:
                            sim = float(np.dot(body_embs[i].squeeze(), body_embs[j].squeeze()))
                            row += f" {sim:6.3f}"
                    row += f"  {body_labels[i]}"
                    print(row)

                # Stats
                sims = []
                for i in range(N):
                    for j in range(i+1, N):
                        if body_embs[i] is not None and body_embs[j] is not None:
                            sims.append(float(np.dot(body_embs[i].squeeze(), body_embs[j].squeeze())))
                if sims:
                    sims.sort()
                    n = len(sims)
                    print(f"\n  Body SigLIP2 stats: min={sims[0]:.3f} p25={sims[n//4]:.3f} "
                          f"p50={sims[n//2]:.3f} p75={sims[3*n//4]:.3f} max={sims[-1]:.3f}")
                    if sims[0] < 0.5:
                        print(f"  ⚠️  MULTIPLE PEOPLE detected (min_sim={sims[0]:.3f} < 0.50)")
                    elif sims[0] < 0.80:
                        print(f"  🟡 SAME person, moderate variance (min_sim={sims[0]:.3f})")
                    else:
                        print(f"  ✅ SAME person, consistent (min_sim={sims[0]:.3f})")

            # Also test face-body similarity if both exist
            if face_rows and N >= 1:
                face_embs = []
                face_labels = []
                for path, score, captured in face_rows:
                    key = path.split("/", 1)[1] if "/" in path else path
                    key = key[len(BUCKET_PREFIX)+1:] if key.startswith(f"{BUCKET_PREFIX}/") else key
                    try:
                        resp = client.get_object(BUCKET_PREFIX, key)
                        data = resp.read(); resp.close(); resp.release_conn()
                        img = Image.open(io.BytesIO(data)).convert("RGB")
                        inputs = processor(images=img, return_tensors="pt").to(device)
                        with torch.no_grad():
                            emb = model.get_image_features(**inputs)
                            emb = emb / emb.norm(dim=-1, keepdim=True)
                        face_embs.append(emb.cpu().numpy())
                        face_labels.append(f"F{captured.strftime('%H:%M')} s={score:.2f}")
                    except Exception:
                        face_embs.append(None)
                        face_labels.append("F err")

                print(f"\n  Face-body SigLIP2 similarity:")
                print(f"  {'':>8}", end="")
                for fl in face_labels:
                    print(f" {fl:>15}", end="")
                print()
                for i in range(N):
                    print(f"  {body_labels[i]:>8}", end="")
                    for j in range(len(face_embs)):
                        if body_embs[i] is not None and face_embs[j] is not None:
                            sim = float(np.dot(body_embs[i].squeeze(), face_embs[j].squeeze()))
                            print(f" {sim:15.3f}", end="")
                        else:
                            print(f" {'ERR':>15}", end="")
                    print()

asyncio.run(run(DEFAULT_IDS if len(sys.argv) < 2 else sys.argv[1:]))
