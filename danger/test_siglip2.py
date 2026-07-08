#!/usr/bin/env python3
"""test_siglip2.py — SigLIP2 zero-shot gender on face crops (PIL input)."""
import asyncio, numpy as np
from sqlalchemy import text
from PIL import Image
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX

IDS = ["d790396b-ef77-4779-8cf9-b2fe6d44f343","a63b766f-e634-48a0-aeba-700f5aa08807","09c53e4f-ce6c-41a4-b7c4-a0bc2ab5347c","8c788ebe-0e89-4ddb-b0c7-ef76b7134a9b","8ea1c658-8511-4368-8fc2-3850d360dafd","c5584e25-960c-48c0-8302-10efca00cbe5","d578e6f7-3e2f-4939-8198-a142708e2851","c24d1dd0-081d-47f8-91f4-e2c1b439f2f1","c5aea7f3-64be-40b1-a849-8bc826c0f201"]

async def run():
    from transformers import pipeline
    pipe = pipeline("zero-shot-image-classification", model="google/siglip2-base-patch16-224", device=0)
    print("SigLIP2 loaded.\n")
    
    for labels in [["a photo of a woman","a photo of a man"], ["woman","man"]]:
        print(f"--- {labels[0][:25]:25s} vs {labels[1][:25]} ---")
        client = get_client(); correct = total = 0
        async with AsyncSessionLocal() as db:
            for pid in IDS:
                r = await db.execute(text("SELECT face_crop_path FROM person_face_embeddings WHERE person_identity_id::text=:pid AND face_crop_path IS NOT NULL ORDER BY face_score DESC LIMIT 3"), {"pid": pid})
                crops = [row[0] for row in r.fetchall()]
                if not crops: continue
                v = {labels[0]:0, labels[1]:0}
                for path in crops:
                    key = path.split("/",1)[1] if "/" in path else path
                    key = key[len(BUCKET_PREFIX)+1:] if key.startswith(f"{BUCKET_PREFIX}/") else key
                    try:
                        resp=client.get_object(BUCKET_PREFIX,key); d=resp.read(); resp.close(); resp.release_conn()
                    except: continue
                    img = Image.open(__import__('io').BytesIO(d)).convert("RGB")
                    if img is None: continue
                    try:
                        result = pipe(img, candidate_labels=labels)
                        winner = result[0]['label']
                        v[winner] += 1
                    except Exception as e: pass
                w,m = v[labels[0]],v[labels[1]]; total += 1
                maj = labels[0] if w>m else (labels[1] if m>w else "Tie")
                if maj==labels[0]: correct+=1
                print(f"  {pid[:12]} {'✓' if maj==labels[0] else '✗'} {maj:>10} (W={w} M={m})")
        print(f"  {correct}/{total} = {round(correct/max(total,1)*100,1)}%\n")

asyncio.run(run())
