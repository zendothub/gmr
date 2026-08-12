# Dual-Quality Browser Streams (Backend)

> Branch: `feature/dual-stream-quality`  
> Date: 2026-08-12

## Problem

Dashboard and multi-camera grids all consumed the **full native** annotated feed
(e.g. 2880×1620). With several tiles open, browser and network bandwidth spiked
unnecessarily.

## Solution

`StreamBroadcaster` publishes **two** MediaMTX paths per camera:

| Variant | MediaMTX path     | Default encode        | Bitrate | API field        | Intended use              |
|---------|-------------------|-----------------------|---------|------------------|---------------------------|
| **LD**  | `cam_<uuid>`      | 1280×720 @ 15 fps     | ~600k   | `webrtc_url`     | Dashboard / grid tiles    |
| **HD**  | `cam_<uuid>_hd`   | height 1024 @ 24 fps* | ~2500k  | `webrtc_url_hd`  | Fullscreen / single cam   |

\* HD width is aspect-preserved from the source, capped by `STREAM_HD_MAX_WIDTH` (1920).  
  Example from 2880×1620 → ~1820×1024.

`webrtc_url` / `hls_url` stay **LD** for backward compatibility.  
HD is additive (`webrtc_url_hd`, `hls_url_hd`, `path_hd`).

## Pipeline

```
RTSP camera (native res)
  → LatestFrameBuffer
  → AI loop (YOLO / face / ReID) — unchanged, full res
  → StreamBroadcaster
       ├─ draw overlays once (source res)
       ├─ resize → FFmpeg LD  → rtsp://MediaMTX/cam_<id>
       └─ resize → FFmpeg HD  → rtsp://MediaMTX/cam_<id>_hd
  → Browser WHEP (WebRTC) / HLS
```

Non-burnin path (`FFmpegPublisher`, `STREAM_PUBLISH_MODE=lowlatency`) scales the
single republish to **LD** dimensions as well.

## Config (`app/config.py` / env)

| Setting | Default | Meaning |
|---------|---------|---------|
| `STREAM_LD_WIDTH` | `1280` | LD width |
| `STREAM_LD_HEIGHT` | `720` | LD height |
| `STREAM_LD_FPS` | `15` | LD frame rate |
| `STREAM_LD_BITRATE` | `600k` | LD H.264 bitrate |
| `STREAM_HD_HEIGHT` | `1024` | HD target height (“1024p”) |
| `STREAM_HD_MAX_WIDTH` | `1920` | HD max width |
| `STREAM_HD_FPS` | `24` | HD frame rate |
| `STREAM_HD_BITRATE` | `2500k` | HD H.264 bitrate |
| `STREAM_PUBLISH_HD` | `True` | Set `False` to publish LD only |
| `STREAM_BURNIN_FPS` | `15` | Legacy alias; LD uses `STREAM_LD_FPS` |
| `STREAM_BITRATE` | `1200k` | Fallback bitrate |

Encoder selection is unchanged: `h264_nvenc` → `h264_videotoolbox` → `libx264`
(`app/utils/device.py::get_ffmpeg_video_codec_args`, per-pipe bitrate).

## API surface

Endpoints that return stream URLs now include optional HD fields:

- `GET /api/v2/cameras/feeds` → `CameraFeedResponse`
- `GET /api/v2/cameras/{id}` / list → `CameraResponse`
- `GET /api/v2/cameras/{id}/polygon-editor`
- `POST /api/cameras/{id}/stream/start` → `StreamEndpointsResponse`
- `GET /api/cameras/{id}/stream/status` → `StreamStatusResponse`

Example:

```json
{
  "webrtc_url": "https://feed…/cam_abc…/whep",
  "hls_url": "https://feed…/cam_abc…/index.m3u8",
  "stream_path": "cam_abc…",
  "webrtc_url_hd": "https://feed…/cam_abc…_hd/whep",
  "hls_url_hd": "https://feed…/cam_abc…_hd/index.m3u8",
  "stream_path_hd": "cam_abc…_hd"
}
```

## Key files

| File | Role |
|------|------|
| `app/modules/ai_runtime/stream_broadcaster.py` | Dual FFmpeg pipes, draw + resize |
| `app/modules/streaming/mediamtx.py` | `camera_path(..., quality)`, dual URLs |
| `app/modules/streaming/ffmpeg_publisher.py` | Non-burnin LD scale |
| `app/modules/streaming/schemas.py` | HD fields on stream responses |
| `app/modules/cameras/schemas.py` | HD fields on camera / feed models |
| `app/config.py` | `STREAM_LD_*` / `STREAM_HD_*` |
| `app/utils/device.py` | Bitrate override on codec args |

## Deploy

```bash
# After merge / checkout of feature/dual-stream-quality
sudo systemctl restart retail-ai.service
# Camera workers load StreamBroadcaster in-process — restart is required.
```

Verify MediaMTX has both paths when a camera is active:

```bash
# Example paths
ffprobe rtsp://localhost:8554/cam_<uuid>
ffprobe rtsp://localhost:8554/cam_<uuid>_hd
```

## Rollback

- Set `STREAM_PUBLISH_HD=false` and restart → LD only (frontend falls back to `webrtc_url`).
- Or revert branch / redeploy previous image.

## Notes

- AI pipeline still runs at full camera resolution; only the **browser publish** is dual-quality.
- Single uvicorn worker and per-camera YOLO rules are unchanged.
- HD encode cost is ~one extra NVENC session per camera; disable via `STREAM_PUBLISH_HD` if GPU is tight.
