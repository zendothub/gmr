# Host Disk Usage & Optimization

> **Host:** GMR Retail Eye on-prem box  
> **Measured:** 2026-08-07  
> **OS volume:** `/dev/sdc2` ext4 mounted at `/`  
> **Related:** [`HOST_CAPACITY_AND_CAMERA_COST.md`](HOST_CAPACITY_AND_CAMERA_COST.md) (RAM/CPU/GPU)

---

## 1. Disk present

| Device | Size | Type | Mount | Role |
|---|---:|---|---|---|
| **ESSENCORE SATA SSD `sdc2`** | **~444–451 GB** usable | ext4 | **`/`** | OS + app + Docker volumes (everything live) |
| `sdc1` | 976 MB | vfat | `/boot/efi` | EFI |
| `sdc3` | 24.6 GB | swap | `[SWAP]` | Swap |
| **WDC `sda` 1.8 TB** | 1.8 TB | BitLocker partition | **not mounted** | Unused capacity |
| **WDC `sdb` 1.8 TB** | 1.8 TB | BitLocker partition | **not mounted** | Unused capacity |

```text
Live capacity that matters today = SSD / only ≈ 444 GB
Extra HDDs (2 × 1.8 TB) exist but are unavailable until unlocked + mounted.
```

### Live fill (2026-08-07)

| Metric | Value |
|---|---:|
| Filesystem size | **444 GB** |
| Used | **~370 GB** |
| Available | **~51 GB** |
| Use % | **~88%** |
| Inodes | 8% used (not an inode crisis) |

**Risk:** under **~50 GB free**, Docker/MinIO/Postgres can hit ENOSPC; AI workers and cleanup jobs fail noisily.

---

## 2. Where the space went (process / component wise)

### 2.1 Top-level map

| Location | ~Size | Owner / process |
|---|---:|---|
| **Docker volume `gmr_minio_data`** | **~335 GB** | **MinIO** (`retaileye_minio` container) |
| `/home/retaileye` | ~26 GB | User cache, IDE, downloads, model caches |
| `/gmr` (app tree) | ~14–15 GB | Backend venv, models, logs, git |
| `/usr` | ~10 GB | System packages |
| `/var` (excl. docker data path) | ~1–2 GB | journal, caches |
| Docker images (minio + mediamtx) | ~230 MB | Images only (data is the volume) |

Almost **all pressure is MinIO object data**, not Python code or GPU models.

### 2.2 MinIO breakdown (dominant)

Docker volume:

```text
Name:        gmr_minio_data
Mountpoint:  /var/lib/docker/volumes/gmr_minio_data/_data
Container:   retaileye_minio  (minio/minio)
Bucket path: /data/retaileye/
```

| Prefix | ~Size | ~Object count | Avg object | Written by |
|---|---:|---:|---:|---|
| **`retaileye/snapshots/`** | **~310–330 GB** | **~270k** | **~1.2 MB** | Event / zone / billing **full-frame snapshots** |
| **`retaileye/crops/`** | **~7.8–8.0 GB** | **~225k** | **~36 KB** | Face + body ReID crops |
| **Total MinIO volume** | **~335 GB** | — | — | — |

Snapshot name pattern (examples):

```text
event_<camera_id>_<YYYYMMDD>_<HHMMSS>_<…>.jpg
```

Observed object date span at measurement: **~2026-07-10 → 2026-08-07** (~4 weeks of continuous write).

### 2.3 Why snapshots are ~330 GB — what / when / why

#### What is stored

| | |
|---|---|
| **Object** | Full camera **frame** encoded as JPEG |
| **Typical size** | ~**1.2 MB** per file |
| **MinIO key** | `retaileye/snapshots/event_<camera_id>_<YYYYMMDD>_<HHMMSS>_<us>_<uuid8>.jpg` |
| **DB link** | `events.snapshot_path` (event list / debug UI evidence image) |
| **Not this prefix** | ReID face/body crops → `retaileye/crops/` (~8 GB only) |

Upload helper: `save_image` / `save_image_async` → MinIO PUT (`app/utils/image_utils.py`).

#### When it is written (code path)

Snapshots are **not** written every AI frame. They are written only when the camera worker persists **rule or zone events** for that frame batch:

```text
AI loop (fps_target)
  → ZoneEventDetector.detect()  +  RuleEvaluator.evaluate()
  → if rule_events or zone_events:
       _persist_events(db, frame, …)
         → save_image_async(frame, SNAPSHOT_DIR="snapshots", prefix=f"event_{camera_id}")
         → Event.snapshot_path = that key  (same path shared by all events in the batch)
```

Source: `CameraWorker._persist_events` in `app/modules/ai_runtime/camera_worker.py`:

```python
snapshot_path = await save_image_async(
    frame, self.settings.SNAPSHOT_DIR, prefix=f"event_{self.camera_id}"
)
```

| Trigger | Event type(s) | Full-frame snapshot? |
|---|---|---|
| Person walks into a zone | `zone_enter` | **Yes** |
| Person leaves a zone | `zone_exit` | **Yes** |
| Dwell hits 30s / 60s / 120s in a zone | `zone_dwell_milestone` | **Yes** |
| Rule fires (billing dwell, line cross, purchase, …) | rule `event_type` | **Yes** |
| New ByteTrack session | `person_entered_view` | **No** — uses **crop** path only |
| Track ends | `person_left_view` | **No** snapshot field |

Zone auto-events come from `ZoneEventDetector` (`zone_enter` / `zone_exit` / milestones **30, 60, 120**).  
Billing / line-crossing come from `RuleEvaluator`.

One `_persist_events` call → **one** JPEG upload; multiple events in the same batch **reuse** that path.

#### Why so much volume (~330 GB)

| Factor | Effect |
|---|---|
| **~270k objects × ~1.2 MB** | ≈ **310–330 GB** arithmetic |
| **2 busy cameras** + counter/entry zones | Frequent enter/exit as people and **staff** move |
| **Track fragmentation** | ByteTrack splits → many short sessions → many zone_enter/exit again |
| **Dwell milestones** | Extra snapshots at 30/60/120s while still in zone |
| **Billing / rules** | Additional events on long counter stays |
| **~11–12 GB/day** write rate | Fills a 444 GB SSD in weeks if retention ≥ ~30d |
| **Purpose of file** | UI/debug **evidence** for events — **not** required for face/body ReID |

```text
Disk crisis driver = snapshots (MB-class, event evidence)
Identity pipeline   = crops     (KB-class, ReID)  ← keep; not the 330 GB problem
```

#### Purpose (product)

| Need | Uses snapshots? |
|---|---|
| Event timeline / “what did the camera see when rule/zone fired” | **Yes** |
| Face match, body ReID, identity create/merge | **No** (uses `crops/`) |
| Purchase count analytics | **No** (uses `billing_interactions` + person ids) |

So MinIO is large because the pipeline **always attaches a full-frame JPEG to every zone/rule event batch**, and those events fire often on live retail CCTV.

### 2.4 App / host (secondary)

| Path | ~Size | Notes |
|---|---:|---|
| `/gmr/gmr/venv` | ~11 GB | Python deps + torch CUDA wheels |
| `/gmr/gmr/logs` | **~2.8 GB** | See log hotspot below |
| `/gmr/gmr/models` | ~633 MB | YOLO / OSNet weights on disk |
| `/home/retaileye/.cache` | ~12 GB | pip/huggingface/torch caches |
| `/home/retaileye/.config` | ~5.3 GB | desktop/app config |
| `/home/retaileye/Downloads` | ~1.9 GB | user files |
| `.vscode` / `.vscode-server` | ~3.1 GB | IDE remote |
| `.insightface` / `.deepface` | ~1.1 GB | model caches |
| `/var/log/journal` | ~769 MB | systemd journal |

#### Log hotspot

| File | ~Size | Risk |
|---|---:|---|
| `/gmr/gmr/logs/retail-ai-error.log` | **~2.3 GB** | Unbounded growth if errors spam |
| Rotated `ai_processing.*.log` | ~48 MB × several | loguru retention helps some streams |

App loguru retention is often **7 days** (`app/main.py` / `logging_config.py`); **systemd/error redirects may not rotate the same way** — watch `retail-ai-error.log`.

### 2.5 What is *not* filling the disk

| Component | Disk footprint |
|---|---|
| Postgres data (on this host path check) | Small on `/` vs MinIO |
| MediaMTX container | Negligible |
| GPU VRAM | Not disk |
| Camera RTSP | Network only |

---

## 3. Current daily disk use (write rate)

Estimates from MinIO totals ÷ active snapshot window (~**28 days**, Jul 10 → Aug 7) at **2 cameras**:

| Stream | ~Daily write | Basis |
|---|---:|---|
| **Event snapshots** | **~11–12 GB/day** | ~330 GB / ~28 d · ~1.2 MB/object |
| **Crops** | **~0.3 GB/day** | ~8 GB / ~28 d · ~36 KB/object |
| **MinIO combined** | **~12 GB/day** | snapshots dominate (~97%) |
| App logs (rough) | **0.05–0.2 GB/day** | spikes if error log floods |
| Code / venv | ~0 steady | only on deploy |

### Steady-state math (if nothing is deleted)

```text
Daily MinIO ≈ 12 GB/day × 2 cams (current behaviour)
30-day keep   ≈ 12 × 30 ≈ 360 GB   ← already near full SSD
60-day keep   ≈ 720 GB              ← does not fit on 444 GB SSD
```

With **~51 GB free** and **~12 GB/day** net growth (if cleanup lags):

```text
Days until ENOSPC ≈ 51 / 12 ≈ 4 days   (order-of-magnitude if cleanup fails)
```

### Built-in cleanup (expected)

| Job | Schedule | Default |
|---|---|---|
| `cleanup_old_storage` | daily **02:00** (`retail-ai-worker`) | **`retention_days=30`** |

```157:167:app/modules/jobs/tasks.py
async def cleanup_old_storage(retention_days: int = 30):
    """Delete snapshots/crops older than the retention period."""
    older_than = utc_now() - timedelta(days=retention_days)
    ...
    removed = await s3_cleanup_old_objects(db, older_than)
```

**Implication:** at ~12 GB/day, a healthy 30-day retention still wants **~360 GB** for MinIO alone — tight on a **444 GB** root disk shared with OS + venv + home. Either **shorter retention**, **move MinIO off `/`**, or **write fewer snapshots**.

If objects are not registered in `storage_objects` (or cleanup errors), MinIO keeps growing past 30 days — verify worker logs for `Storage cleanup job removed…`.

---

## 4. How to optimize

### 4.1 P0 — stop the bleeding (same day)

| Action | Expected win |
|---|---|
| **Confirm worker cleanup runs** | `journalctl -u retail-ai-worker` / `logs/ai_processing.log` for storage cleanup |
| **Shorten retention** (e.g. 30 → **7–14 days** for snapshots) | Free **tens–hundreds of GB** if old objects exist |
| **Truncate / rotate `retail-ai-error.log`** | Free **~2 GB** immediately |
| **Do not delete MinIO files by hand** while cams run without DB consistency | Prefer job / `storage_objects`-aware cleanup (see crop lifecycle docs) |

Example: run cleanup with a tighter window (if your deploy exposes a one-shot or you temporarily change the job arg — prefer a controlled script over random `rm`):

```bash
# Check free space before/after
df -h /

# Inspect MinIO volume size
docker system df -v | sed -n '1,40p'

# App error log (safe if service restarted after truncate)
sudo truncate -s 0 /gmr/gmr/logs/retail-ai-error.log   # only if you accept losing that log
# or rotate:
sudo mv /gmr/gmr/logs/retail-ai-error.log /gmr/gmr/logs/retail-ai-error.log.old
sudo systemctl restart retail-ai.service
```

### 4.2 P1 — cut daily write rate (best long-term on small SSD)

| Action | Effect on disk |
|---|---|
| **Fewer event snapshots** | Snapshots are **~12 GB/day**; crops only ~0.3 GB/day — optimize events first |
| Lower snapshot JPEG quality / resolution (if configurable) | Smaller MB/event |
| Avoid duplicate snapshot writes per rule fire | Fewer objects |
| Keep crops for ReID; expire **snapshots** faster than crops | Identity quality vs dashboard thumbs tradeoff |
| Burn-in / debug dumps off disk | Avoid extra local writes |

Rule of thumb:

```text
Disk crisis driver = snapshots (MB-class)
ReID driver         = crops (KB-class)  ← usually keep longer
```

### 4.3 P2 — move data off the 444 GB SSD

| Action | Effect |
|---|---|
| Unlock + mount one **1.8 TB** HDD (BitLocker today) | Real capacity for MinIO |
| Relocate Docker volume `gmr_minio_data` to that mount | Frees **~335 GB** on `/` |
| Or point MinIO bind-mount at `/mnt/data/minio` | Cleaner ops than anonymous volume on `/` |

Until HDDs are mounted, **they do not help** — disk full means SSD full.

### 4.4 P3 — hygiene

| Action | Win |
|---|---|
| Clear old `/home/retaileye/.cache` (pip/HF) if safe | up to ~12 GB |
| Prune unused Docker images (little here) | ~0.2 GB |
| Cap journal: `SystemMaxUse=500M` | journal growth |
| Ensure loguru retention applies to all sinks | prevent multi-GB error logs |

```bash
# Optional cache cleanup (review first)
du -sh /home/retaileye/.cache/*
# docker
docker system df
```

### 4.5 Target layout (recommended)

```text
SSD /          : OS + app + venv + postgres (if local) + logs   (~80–120 GB steady)
HDD /mnt/minio : MinIO bucket data                              (TB-class growth OK)
Retention      : snapshots 7–14d · crops 14–30d (product choice)
Daily budget   : aim < 5 GB/day snapshots on small SSD, or unlimited on HDD
Alert          : df -h /  when Avail < 40 GB
```

---

## 5. Quick commands (ops)

```bash
# Free space
df -h /
df -ih /

# Docker / MinIO volume
docker system df -v
docker ps --filter name=minio
docker exec retaileye_minio du -sh /data/retaileye/* 2>/dev/null

# Host heavy paths
du -xh --max-depth=1 /home/retaileye /gmr /var 2>/dev/null | sort -hr | head
du -sh /gmr/gmr/logs/* 2>/dev/null | sort -hr | head

# Disks present but unused
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL
```

---

## 6. Bottom line

| Question | Answer |
|---|---|
| How much disk is present? | **~444 GB** live SSD; **+2×1.8 TB** unused (BitLocker, unmounted) |
| How much used? | **~370 GB (88%)** |
| Who uses it? | **MinIO ~335 GB** (snapshots ~310 GB, crops ~8 GB) |
| Why snapshots ~330 GB? | Full-frame JPEG on every **zone/rule event** batch (`_persist_events`); ~1.2 MB × ~270k; not used for ReID |
| When written? | `zone_enter` / `zone_exit` / dwell 30·60·120 / billing+rules — **not** every frame; enter/leave view use crops or no snap |
| Daily use now? | **~12 GB/day** MinIO (almost all event snapshots) @ 2 cams |
| Why “too full”? | Snapshots on small root SSD; HDDs not in play; 30d retention ≈ full disk |
| Optimize first? | **Shorter snapshot retention + fix cleanup + rotate error log**; then **move MinIO to HDD**; then reduce snapshot write rate |

---

## Related

- Host RAM/CPU/GPU capacity: [`HOST_CAPACITY_AND_CAMERA_COST.md`](HOST_CAPACITY_AND_CAMERA_COST.md)
- MinIO crop lifecycle / deferred delete: [`CROP_LIFECYCLE_AND_MINIO_FIXES.md`](CROP_LIFECYCLE_AND_MINIO_FIXES.md)
- Worker jobs: `app/worker.py`, `app/modules/jobs/tasks.py` (`cleanup_old_storage`)
- Architecture memory: [`../CONTEXT.md`](../CONTEXT.md)
