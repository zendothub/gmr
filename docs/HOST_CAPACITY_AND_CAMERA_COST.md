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

**Note:** Most model VRAM is a **fixed cost per AI process**, not linear per camera. YOLO+ByteTrack is **per camera**. NVENC burn-in is **~0.3 GB GPU + ~0.3 GB RAM per camera** (encoder only — see §6–7 for full per-cam CPU/GPU-time cost and +10 bottlenecks).

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

## 6. Resources needed for each extra camera

Do **not** treat “~0.3 GB GPU + ~0.3 GB RAM” as the full cost of one camera. That number is **only the optional NVENC burn-in stream**.

### 6.1 Fixed vs per-camera (already loaded @ 2 cams)

| Cost type | What | GPU VRAM | Host RAM | CPU / GPU time |
|---|---|---:|---:|---|
| **Fixed (once)** | InsightFace, SigLIP2, OSNet, YOLO-Pose, Torch CUDA context | ~**3.0–3.2 GB** | ~**4 GB** in uvicorn | shared |
| **Fixed (jobs)** | `app.worker` (dedup etc.) | ~**0.7 GB** | ~**2.5–2.7 GB** | low |
| **Per camera (required)** | YOLO + ByteTrack instance | ~**0.05–0.15 GB** (weights/state order) | tracker + frame buffers (100s MB class) | **every AI frame** |
| **Per camera (required)** | RTSP decode (OpenCV / FFmpeg pull) | 0 | decode buffers | **CPU** linear in N × resolution |
| **Per camera (required)** | AI pipeline @ `fps_target` (default **5**) | 0 extra weights | small | **GPU time**: YOLO + full-frame face + ReID + identity |
| **Per camera (optional)** | NVENC burn-in 2880×1620 @ 15 fps | **~0.3 GB** | **~0.3 GB** | NVENC session + some CPU feed |

```text
Full stack (models)     ≈ fixed once the process is up
+ each camera           ≈ decode CPU + YOLO/ByteTrack + (N × 5) AI frames/s of shared GPU work
+ each burn-in stream   ≈ +0.3 GB GPU + +0.3 GB RAM   ← only this is the “0.3 / 0.3” line
```

### 6.2 Why free VRAM ÷ 0.3 ≠ max cameras

| Naïve math | Why it fails |
|---|---|
| Free ~6.9 GB VRAM ÷ 0.3 ≈ “23 cams” | 0.3 is **encoder only**; does not pay for decode or AI frames |
| GPU util ~10% at 2 cams → “room for 10×” | Util is **idle time**, not free camera slots; work ≈ **N × fps** |
| RAM free ~17 GB ÷ 0.3 ≈ “many cams” | Same — burn-in RAM only; decode + tracks + DB still grow |

### 6.3 Rough marginal budget per extra camera (RAM / GPU / CPU only)

| Item | Burn-in **ON** | Burn-in **OFF** |
|---|---|---|
| GPU VRAM | ~**0.3 GB** (NVENC) + small YOLO | small YOLO only |
| Host RAM | ~**0.3 GB** (ffmpeg) + buffers/tracks | buffers/tracks only |
| CPU | decode + feed encoder + pipeline | decode + pipeline |
| GPU compute | +`fps_target` frames/s on shared GPU | same |
| NVENC slots | **1 session** (desktop ~**3** concurrent hard limit) | 0 |

AI default: **`fps_target = 5`**.  
Total AI pressure ≈ **N × 5** frames/s (e.g. 12 cams → ~**60** pipeline frames/s through one process / one GPU).

---

## 7. Bottleneck if you add **+10 cameras** (total **12**)

Baseline today: **2 cams**. Scenario: add **10 more** → **12** total, still current architecture (single uvicorn, per-cam YOLO+ByteTrack, shared IF/SigLIP/OSNet), **`fps_target=5`**. Disk/MinIO ignored.

### 7.1 Ordered bottlenecks (what breaks first)

| Order | Bottleneck | What happens at total ~12 |
|---:|---|---|
| **1** | **NVENC burn-in (if left ON)** | Desktop 4070 Ti ~**3** concurrent full h264_nvenc sessions. Cams 4–12 cannot all keep full 2880×1620 burn-in even if VRAM still has free MiB. |
| **2** | **GPU compute time (AI path)** | 12 × 5 = **~60 AI frames/s** of YOLO + full-frame SCRFD + ReID + identity on **one** GPU. Util jumps; **p95 FPS drops below 5** under people; latency/jitter before OOM. |
| **3** | **CPU RTSP decode** | 12 full-res pulls. Load avg leaves the “~20–30% of 24 threads” comfort zone; threads fight decode vs pipeline. |
| **4** | **Single-process architecture** | All `CameraWorker`s in one `--workers 1` process: GIL, thread count, shared identity path, DB lock contention. |
| **5** | **Per-cam YOLO + ByteTrack memory/state** | Extra VRAM/RAM beyond 0.3; grows with N but usually **after** (1)–(3). |
| **6** | **VRAM OOM** | Unlikely as the *first* wall if burn-in is capped; free ~6.9 GB is mostly unused **weights budget**, not free **compute**. |
| **7** | **Identity / DB quality** | More tracks → more creates/merges/advisory lock time; product truth degrades even if process stays up. |

### 7.2 With burn-in ON vs OFF at +10

| Mode | Total N | Expected outcome |
|---|---:|---|
| Burn-in **ON** every cam | 12 | **Fails early** on NVENC sessions (~3). Not a VRAM-arithmetic problem. |
| Burn-in **OFF** (or ≤2–3 full streams) | 12 | VRAM/RAM may still fit; **GPU time + CPU decode** are the walls. Expect FPS shortfall and identity lag on busy floor — **stretch / not SLA**. |
| Honest current-arch band | **4–6** (burn-in limited) | Sustainable without re-arch |
| +10 without re-arch | **12** | **Not recommended** as sold capacity |

### 7.3 What is *not* the bottleneck for +10

| Not the first wall | Why |
|---|---|
| “Only 0.3 GB free needed × 10” | Misreads burn-in line as full cam cost |
| Empty VRAM slots alone | Compute and NVENC saturate first |
| Disk / MinIO (if ops deletes/shifts data) | Out of scope for this RAM/GPU/CPU analysis |

### 7.4 Cost at total 12 (server-only, if it ran)

| N | ₹/cam/mo (₹17k / N) |
|---:|---:|
| 12 | **≈ ₹1,417** |

Economics look fine; **technical SLA does not** on current pipeline without batching + stream caps.

---

## 8. Bottom line

| Question | Answer |
|---|---|
| Headroom now? | CPU/GPU **util** free; **NVENC + decode + AI time + architecture** limit cams first |
| What does each extra cam need? | Decode CPU + YOLO/ByteTrack + **N×5 FPS** shared GPU work; **+0.3/0.3 only if burn-in ON** |
| Add safely today? | **+1** → total **3** → **~₹5,667/cam/mo** |
| Best near-term density? | Total **4–6** (limit burn-in) → **₹4,250–₹2,833/cam/mo** |
| +10 cams (total 12)? | **Bottleneck:** NVENC (if on) → GPU AI time → CPU decode → single process — **not** free VRAM ÷ 0.3 |
| Path to 24–40? | Shared batch inference, cap full streams ≤5, drop default per-cam NVENC |

---

## Related

- Main backend README: [`../README.md`](../README.md)
- Cross-session architecture memory: [`../CONTEXT.md`](../CONTEXT.md)
- Host-level capacity critique (workspace): `/gmr/CAPACITY_CRITIQUE_ZORIK_96CAM.md`
