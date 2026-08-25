# Host Capacity & Camera Cost

> **Host:** GMR Retail Eye on-prem box  
> **Measured:** 2026-08-10 (storage refreshed; compute baseline 2026-08-07)  
> **Server OpEx used for ₹ tables:** ₹17,000 / month (see §3.1 — why ₹17k)  
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

### Storage (disks)

| Device | Type | Model | Present | Used | Free |
|--------|------|-------|--------:|-----:|-----:|
| **sdc** | **SSD** | ESSENCORE SATA SSD | 476.9 GB | ~80 GB (19% on `/` 444 GB) | ~342 GB on `/` |
| **sda** | HDD | WDC WD20EZRZ-22Z5HB0 | 1.8 TB | — | unused (BitLocker partition, unmounted) |
| **sdb** | HDD | WDC WD20EZRZ-00Z5HB0 | 1.8 TB | — | unused (BitLocker partition, unmounted) |

OS + app + MinIO live on **sdc** only. HDDs are present capacity, not in the live path until mounted/unlocked. Monitor crop lifecycle / cleanup so `/` does not fill.

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

### 3.1 How ₹17,000 is built (host setup + wifi — **no cameras**)

₹17k/mo is **infra only**: amortize this box’s hardware + run power + wifi/ISP.  
**Cameras, lenses, PoE, NVR ports are excluded.**

#### A) Host setup (one-time CAPEX → monthly)

Hardware present on this GMR box (India street-price order-of-magnitude; replace with PO when available):

| Part | Spec (this host) | Est. one-time ₹ |
|---|---|---:|
| CPU | Intel i9-10920X (12c/24t) | ~45,000 |
| GPU | RTX 4070 Ti 12 GB | ~80,000 |
| RAM | ~32 GB | ~10,000 |
| SSD | ESSENCORE ~477 GB (OS/app) | ~4,000 |
| HDD | 2× WD 1.8 TB (present; optional data) | ~12,000 |
| Mobo + PSU + case + cabling | X299-class build | ~30,000 |
| **Host CAPEX total** | | **~₹1,81,000** |

```text
Host amort / mo  =  CAPEX / months
                 =  181000 / 36     ≈  ₹5,030 / mo   (3-year life)
```

| Life | Monthly host amort |
|---|---:|
| 24 months | ~₹7,540 |
| **36 months (used here)** | **~₹5,030** |
| 48 months | ~₹3,770 |

#### B) Monthly run cost (no cameras)

| Line | What | Est. ₹/mo | Notes |
|---|---|---:|---|
| **Host amort** | §A, 36-mo | **~5,000** | depreciation of box only |
| **Power** | CPU+GPU+disks 24×7 | **~3,000** | ~350–450 W avg × ~₹8–10/kWh × 720 h |
| **Wifi / ISP** | store uplink for RTSP + dashboard | **~2,000** | broadband / wifi bill class |
| **Spares / UPS share / minor IT** | PSU fan, disk risk, small UPS share | **~2,000** | keep box online |
| **Site / rack / ops buffer** | floor space, remote hands buffer | **~5,000** | on-prem overhead (not cam install) |
| **Total ≈** | | **~₹17,000** | |

```text
₹17,000 / mo  ≈  host amort (~5k)
              +  power (~3k)
              +  wifi/ISP (~2k)
              +  spares/UPS (~2k)
              +  site/ops buffer (~5k)

NO camera BOM in this total.
```

#### C) Explicitly **out** of ₹17k

- IP cameras, lenses, mounts  
- PoE switch / camera cabling  
- Per-camera licenses sold as product SKU  
- NOC staff salary, multi-site cloud  

#### D) If you only want “box + wifi” (tighter)

| Scope | ₹/mo |
|---|---:|
| Host amort + power + wifi only | ~5k + 3k + 2k = **~₹10,000** |
| **Full infra planning (doc default)** | **~₹17,000** |

Use **₹10k** numerator if finance rejects site/spares buffer; all ₹/cam rows scale by `10000/17000`.

### 3.2 How ₹/cam is calculated

```text
monthly_opex          = 17000          # from §3.1 (or 10000 tight)
₹/cam/mo (full host)  = monthly_opex / N
₹/cam/mo (70% sold)   = monthly_opex / (0.70 × N)
```

| Symbol | Meaning |
|---|---|
| `monthly_opex` | Host+wifi(+buffer) — **fixed**; does not grow with N |
| `N` | Cameras the host can honestly run (capacity band) |
| full host | Split entire opex across all N cams |
| 70% billable | Cost on sold cams only |

**Examples (₹17k opex, no camera hardware in numerator):**

| N | Full ₹/cam/mo | @70% billable |
|---:|---:|---:|
| 2 (live now) | 17000/2 = **₹8,500** | — |
| 3 (safe + burn-in) | 17000/3 = **₹5,667** | ₹8,095 |
| 40 (plan after re-arch) | 17000/40 = **₹425** | ₹607 |
| 96 (report — rejected) | 17000/96 = **₹177** | cheap only if N is real |

Replace any line in §3.1 with real invoices; keep the same formulas.

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

## 8. How to optimize with current resources

No new hardware. Goal: free NVENC/GPU/CPU headroom and fit more cameras on this 4070 Ti box. Live baseline at doc time: **2 cams**, full burn-in **2880×1620 @ 15 fps**, AI often **`fps_target=5`**, GPU util ~**8–12%**.

### 8.1 P0 — free the most headroom (do first)

| Action | Effect |
|---|---|
| **Burn-in only when viewing** | Set `burnin_enabled=false` on cams not watched → drop ~**0.3 GB GPU** + **1 NVENC slot** each |
| **Cap concurrent full streams ≤2–3** | Do not leave every cam on full NVENC 2880×1620 @15 if unused |
| **Stream mode `copy` when no burn-in** | `STREAM_PUBLISH_MODE=copy` — remux only, cheaper than `lowlatency` re-encode |
| **Lower burn-in cost if stream must stay on** | Downscale feed and/or `STREAM_BURNIN_FPS` **15→8** → less NVENC + CPU |

### 8.2 P1 — more cameras on the same box

| Action | Effect |
|---|---|
| Keep AI **`fps_target=5`** on busy cams (entry / counter) | Preserve tracking + identity quality |
| Quiet / aisle cams **`fps_target=3`** | Cuts pipeline load ~**40%** on those cams |
| Add new cams **with burn-in off** by default | Path to total **4–6** without NVENC wall |
| Do **not** raise multi-cam AI to 10 FPS | Config default may be 10 for light N; multi-cam stay **5** (or 3) |

### 8.3 P2 — use idle CPU/GPU better (stability / quality)

| Action | Effect |
|---|---|
| Keep **single** uvicorn `--workers 1` | Mandatory (in-process camera state) |
| Keep inference pool **`MAX_WORKERS` ~10–12** | Avoid one cam’s ReID starving another’s YOLO |
| Per-camera YOLO + ByteTrack | Do not share YOLO instances (tracker corruption) |
| Mount extra disks for MinIO when needed | Ops only — not a compute unlock |

### 8.4 P3 — later (real scale, engineering)

| Action | Effect |
|---|---|
| Shared **batch** inference (weights once; ByteTrack state per cam) | Unlocks 10s of cams on ~3 GB VRAM floor |
| Face cadence **≠** YOLO cadence (face less often) | Primary GPU-time lever |
| No default per-cam NVENC; ≤**5** full WebRTC viewers | Removes encode wall |
| Dynamic FPS on idle cameras | Night / empty-floor headroom |

Path after P3: SLA **~24** / plan **~40** (see §4). Not a config toggle today.

### 8.5 Practical target on this box

```text
Now:     2 cams + burn-in ON              → fine, GPU util ~10%
Better:  burn-in OFF by default           → free NVENC + ~0.6 GB GPU @2 cams
Scale:   +2–4 cams @5 FPS, burn-in on demand → total 4–6
Avoid:   10 cams all burn-in @15 FPS full res
```

**Biggest win today:** stop always-on full-res burn-in; stream on demand. That unlocks the next cameras more than “CPU is only 30% / GPU 10%.”

### 8.6 Config knobs (reference)

| Knob | Where | Typical optimize value |
|---|---|---|
| `burnin_enabled` | per camera | `false` unless someone is watching |
| `fps_target` | per camera | **5** busy · **3** quiet |
| `STREAM_PUBLISH_MODE` | env / config | `copy` without burn-in; `lowlatency` if re-encode needed |
| `STREAM_BURNIN_FPS` | env / config | **8–15** (lower = cheaper encode) |
| `STREAM_BITRATE` | env / config | e.g. `1200k` — do not raise without need |
| `MAX_WORKERS` | env / config | **10–12** inference threads |
| uvicorn `--workers` | systemd | **1** only |

Restart after env changes: `sudo systemctl restart retail-ai.service`.

---

## 9. Bottom line

| Question | Answer |
|---|---|
| Headroom now? | CPU/GPU **util** free; **NVENC + decode + AI time + architecture** limit cams first |
| What does each extra cam need? | Decode CPU + YOLO/ByteTrack + **N×5 FPS** shared GPU work; **+0.3/0.3 only if burn-in ON** |
| Optimize first? | **Burn-in off by default** · stream on demand · quiet cams `fps_target=3` · `STREAM_PUBLISH_MODE=copy` |
| Add safely today? | **+1** → total **3** → **~₹5,667/cam/mo** |
| Best near-term density? | Total **4–6** (limit burn-in) → **₹4,250–₹2,833/cam/mo** |
| +10 cams (total 12)? | **Bottleneck:** NVENC (if on) → GPU AI time → CPU decode → single process — **not** free VRAM ÷ 0.3 |
| Path to 24–40? | Shared batch inference, cap full streams ≤5, drop default per-cam NVENC |

---

## Related

- Main backend README: [`../README.md`](../README.md)
- Cross-session architecture memory: [`../CONTEXT.md`](../CONTEXT.md)
- Host-level capacity critique (workspace): `/gmr/CAPACITY_CRITIQUE_ZORIK_96CAM.md`
