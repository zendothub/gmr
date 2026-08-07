# Host Capacity & Camera Cost

> **Host:** GMR Retail Eye on-prem box  
> **Measured:** 2026-08-07  
> **Server OpEx used for ₹ tables:** ₹17,000 / month  
> **Live cameras at measurement:** 2 (full NVENC burn-in @ 2880×1620)

---

## 1. Hardware — present vs used

### CPU

| | |
|---|---|
| **Present** | Intel Core i9-10920X · 12 cores / 24 threads @ 3.5–4.8 GHz |
| **Used (2 cams)** | Load avg ~4.5–6.7 (~19–28% of 24 threads) |
| **Headroom** | High — CPU is not the first wall at low camera count |

### RAM

| | |
|---|---|
| **Present** | ~31–33 GB (+ 24 GB swap) |
| **Used** | ~14 GB (~42%) |
| **Available** | ~19 GB |

| Process | RSS (approx) |
|---|---:|
| `uvicorn` (API + AI, 2 cams) | ~4.3 GB |
| `python -m app.worker` | ~2.7 GB |
| `ffmpeg` NVENC ×2 | ~0.32 GB each |
| OS / Postgres / MinIO / desktop | remainder |

### GPU

| | |
|---|---|
| **Present** | NVIDIA GeForce RTX 4070 Ti · **12 282 MiB** · 285 W TDP |
| **Used (2 cams)** | **~5 086 MiB** (~41%) · util often **~8–12%** · ~39 W |
| **Free** | **~6 900 MiB** |

| Process | GPU VRAM |
|---|---:|
| `uvicorn` (InsightFace + SigLIP2 + YOLO×2 + OSNet + overhead) | ~3 148 MiB |
| `app.worker` | ~718 MiB |
| `ffmpeg` h264_nvenc cam 1 | ~310 MiB |
| `ffmpeg` h264_nvenc cam 2 | ~310 MiB |

**Note:** Most model VRAM is a **fixed cost per AI process**, not linear per camera. YOLO+ByteTrack is **per camera**. NVENC burn-in is **~0.3 GB GPU + ~0.3 GB RAM per camera**.

### Storage

| Volume | Present | Used | Free |
|---|---:|---:|---:|
| OS `/` (`sdc2`) | ~444 GB | ~368 GB (88%) | ~53 GB |
| Extra disks `sda` / `sdb` | 2× 1.8 TB | not mounted | unused |

Root disk is tight for unbounded MinIO growth — monitor crop lifecycle / cleanup jobs.

---

## 2. Architecture facts that limit scaling

| Fact | Implication |
|---|---|
| Single `uvicorn --workers 1` | All camera workers share one process (required for in-process state) |
| YOLO + ByteTrack **per camera** | Tracker isolation; small extra VRAM/RAM per cam |
| InsightFace / SigLIP2 / OSNet **shared** | VRAM floor ~3 GB once models load |
| Full-res NVENC burn-in **per camera** | ~0.3 GB GPU each; desktop NVENC concurrency typically **~3** hard sessions |
| AI `fps_target` default **5** | Total pipeline pressure ≈ N × fps |

GPU util is often low at 2 cams → **underused compute**, not VRAM-starved. Future walls: **NVENC, decode, face path, identity/DB** — not “fill empty VRAM slots.”

---

## 3. Current cost (2 cameras)

| Metric | Value |
|---|---:|
| Server OpEx | **₹17,000 / mo** |
| Cameras live | **2** |
| **Cost per camera (full host)** | **₹8,500 / mo** |
| Shared AI stack | ~3.1 GB GPU + ~4.3 GB RAM |
| Per-cam burn-in | ~0.3 GB GPU + ~0.3 GB RAM each |

Formula:

```text
₹/cam/mo = 17000 / N
```

---

## 4. How many cameras can you add (current architecture)

| Scenario | Total cams | Add now | Constraint |
|---|---:|---:|---|
| **A — Safe + full NVENC burn-in** | **3** | **+1** | NVENC ~3 concurrent sessions |
| **B — Balanced (burn-in off / ≤ few full streams)** | **4–6** | **+2–4** | Keep `fps_target=5` |
| **C — Stretched** | **~8** | **+6** | `fps_target=3`; not SLA quality |
| **D — After re-arch** (shared batch, no per-cam NVENC, ≤5 full streams) | **24 SLA / 40 plan** | +22 / +38 | Engineering work required |

### Practical recommendation (today)

- **Safest add:** **+1 camera** (total **3**) with burn-in still on.
- **Best density without re-arch:** total **4–6** if per-cam burn-in is disabled (or only a few cams stream full).
- **Do not sell 10+ or 96** on this SKU without measured re-architecture.

Validated planning bands (see also `CAPACITY_CRITIQUE` notes on host):

| Band | Cameras | Use |
|---|---:|---|
| SLA (dense pharmacy + RE identity) | **24** | Contract after re-arch |
| Plan (~5 FPS AI, lean streams) | **40** | Capacity + ₹ planning after re-arch |
| Peak lab | **48** | Experiment only |
| Report-style 96 | **reject** | Not validated on 4070 Ti 12 GB |

---

## 5. Cost per camera at capacity

| N cams | You add vs today | ₹/cam/mo (full) | ₹/cam @70% billable |
|---:|---:|---:|---:|
| **2** (now) | 0 | **₹8,500** | — |
| **3** (safe + burn-in) | +1 | **₹5,667** | ₹8,095 |
| **4** | +2 | **₹4,250** | ₹6,071 |
| **6** (balanced max) | +4 | **₹2,833** | ₹4,048 |
| **8** (stretch) | +6 | **₹2,125** | ₹3,036 |
| **24** (SLA after re-arch) | +22 | **₹708** | ₹1,000 |
| **40** (plan after re-arch) | +38 | **₹425** | ₹607 |

Optional light add-ons (order-of-magnitude): bandwidth ~₹2k + storage ~₹0.5k → at N=40 ≈ **₹488/cam/mo** all-in.

---

## 6. Marginal resource cost per extra camera

| Item | Approx cost |
|---|---|
| NVENC burn-in (2880×1620 @ 15 fps) | **~0.3 GB GPU + ~0.3 GB RAM** |
| YOLO + ByteTrack instance | small VRAM + tracker state |
| Shared models (IF / SigLIP2 / OSNet) | **0** extra if already loaded |
| AI pipeline time | ~`fps_target` frames/s of YOLO + full-frame face + ReID + identity |

---

## 7. Bottom line

| Question | Answer |
|---|---|
| Headroom now? | CPU/GPU **util** free; **NVENC + architecture** limit cams first |
| Add safely today? | **+1** → total **3** → **~₹5,667/cam/mo** |
| Best near-term density? | Total **4–6** (limit burn-in) → **₹4,250–₹2,833/cam/mo** |
| Path to 24–40? | Shared batch inference, cap full streams ≤5, drop default per-cam NVENC |

---

## Related

- Main backend README: [`../README.md`](../README.md)
- Cross-session architecture memory: [`../CONTEXT.md`](../CONTEXT.md)
- Host-level capacity critique (workspace): `/gmr/CAPACITY_CRITIQUE_ZORIK_96CAM.md`
