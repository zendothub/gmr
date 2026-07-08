#!/usr/bin/env python3
"""
test_deepface.py — test DeepFace gender classifier on known female persons.
"""

import asyncio
import sys
import cv2
import numpy as np

from sqlalchemy import text
from app.core.db.session import AsyncSessionLocal
from app.modules.storage.minio_client import get_client, BUCKET_PREFIX
from deepface import DeepFace

FEMALE_IDS = [
    "d790396b-ef77-4779-8cf9-b2fe6d44f343",
    "a63b766f-e634-48a0-aeba-700f5aa08807",
    "09c53e4f-ce6c-41a4-b7c4-a0bc2ab5347c",
    "8c788ebe-0e89-4ddb-b0c7-ef76b7134a9b",
    "8ea1c658-8511-4368-8fc2-3850d360dafd",
    "c5584e25-960c-48c0-8302-10efca00cbe5",
    "d578e6f7-3e2f-4939-8198-a142708e2851",
    "c24d1dd0-081d-47f8-91f4-e2c1b439f2f1",
    "c5aea7f3-64be-40b1-a849-8bc826c0f201",
]

async def test():
    client = get_client()
    correct = 0
    total = 0
    
    async with AsyncSessionLocal() as db:
        for pid in FEMALE_IDS:
            r = await db.execute(text(
                "SELECT face_crop_path FROM person_face_embeddings WHERE person_identity_id::text=:pid AND face_crop_path IS NOT NULL ORDER BY face_score DESC LIMIT 5"
            ), {"pid": pid})
            crops = [row[0] for row in r.fetchall()]
            if not crops:
                print(f"  {pid[:12]}: no crops")
                continue
            
            votes = {"Woman": 0, "Man": 0}
            errors = 0
            for path in crops:
                key = path.split("/", 1)[1] if "/" in path else path
                if key.startswith(f"{BUCKET_PREFIX}/"): key = key[len(BUCKET_PREFIX)+1:]
                try:
                    resp = client.get_object(BUCKET_PREFIX, key)
                    data = resp.read(); resp.close(); resp.release_conn()
                except:
                    errors += 1
                    continue
                
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    errors += 1
                    continue
                
                try:
                    # DeepFace expects BGR image (OpenCV format)
                    result = DeepFace.analyze(img_path=frame, actions=['gender'], enforce_detection=False, silent=True)
                    gender = result[0]['dominant_gender']  # "Man" or "Woman"
                    votes[gender] += 1
                except Exception as e:
                    errors += 1
            
            woman_votes, man_votes = votes.get("Woman", 0), votes.get("Man", 0)
            majority = "Woman" if woman_votes > man_votes else ("Man" if man_votes > woman_votes else "Tie")
            ok = "✓" if majority == "Woman" else "✗"
            
            shown = sum(votes.values()) + errors
            if shown > 0:
                total += 1
                if majority == "Woman": correct += 1
            
            print(f"  {pid[:12]}  {ok}  DeepFace={majority:>6}  (W={woman_votes} M={man_votes}  err={errors}/{len(crops)})")
    
    acc = round(correct / max(total, 1) * 100, 1)
    print(f"\nDeepFace accuracy: {correct}/{total} = {acc}%")
    return correct, total

if __name__ == "__main__":
    asyncio.run(test())
