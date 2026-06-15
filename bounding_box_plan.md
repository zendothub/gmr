# Live Stream Bounding Box Overlay Plan

This plan details how to display bounding boxes for tracked persons on top of the live WebRTC/HLS camera stream.

---

## Architectural Comparison

We have two distinct architectural approaches for implementing live bounding boxes. **We recommend Option 1 (WebSocket + Client-side Overlay)** because it avoids costly video re-encoding on the backend, allowing the server to scale to multiple camera streams easily.

### Comparison of Approaches

| Criteria | Option 1: WebSocket + Canvas/SVG Overlay (Recommended) | Option 2: Server-side Burn-in (FFmpeg Re-encoding) |
| :--- | :--- | :--- |
| **Backend CPU Cost** | **Extremely Low** (JSON serialized coordinate broadcasts only) | **Extremely High** (FFmpeg H.264 re-encoding of every frame) |
| **Video Quality** | Pristine, native camera quality | Reduced quality or high CPU load to maintain high bitrate |
| **Sync Accuracy** | Sub-second sync matching WebRTC latency (~150-300ms drift) | Perfect frame-by-frame sync |
| **UX Toggleability** | Turn overlay on/off instantly on client without restarting stream | Must reload player and connect to a different video stream |
| **Interactive Info** | Supports tooltips, hover effects, and clicking on boxes | Purely visual pixels (no UI interaction) |

---

## Open Questions / Decisions

1. **Option Selection:** Do you prefer **Option 1 (recommended)** for scaling performance, or is **Option 2** required due to a constraint that demands perfect frame synchronization?
2. **Overlay Display Style:** For the visual bounding box overlay style, should we show:
   - Bounding boxes + local Track IDs (e.g. `ID: 4`)?
   - Bounding boxes + ReID Resolved Person name/ID?
   - Bounding boxes + Demographic guesses (e.g. `Male, 25-34` if demographics are enabled)?

---

## Proposed Changes (Assuming Option 1)

### Backend Components (gmr-be)

1. **New Connection Manager (`telemetry.py`):**
   * Create [telemetry.py](file:///Users/zulqarnain/GitProjects/bcss/gmr-poc/gmr-be/app/modules/ai_runtime/telemetry.py).
   * Holds a registry of active WebSockets per camera and broadcasts coordinate updates.

2. **FastAPI Route (`router.py`):**
   * Modify [router.py](file:///Users/zulqarnain/GitProjects/bcss/gmr-poc/gmr-be/app/modules/ai_runtime/router.py) to add a WebSocket endpoint: `@router.websocket("/{camera_id}/telemetry")`.
   * Accept, manage, and clean up connections.

3. **Frame Processing Loop Broadcast (`camera_worker.py`):**
   * Modify [camera_worker.py](file:///Users/zulqarnain/GitProjects/bcss/gmr-poc/gmr-be/app/modules/ai_runtime/camera_worker.py).
   * Inside `_process_frame`, serialize each track's normalized bounding box (float percentage coords `0.0` to `100.0`) and broadcast them asynchronously to the telemetry manager.

---

### Frontend Components (gmr-frontend)

1. **Telemetry React Hook (`useTelemetry.ts`):**
   * Create [useTelemetry.ts](file:///Users/zulqarnain/GitProjects/bcss/gmr-poc/gmr-frontend/src/features/live/services/useTelemetry.ts) hook that binds to the camera-specific WebSocket endpoint.
   * Maintains local state of active bounding boxes, adding simple decay buffers to avoid coordinates flickering.

2. **Overlay Rendering & Controls (`StreamPlayer.tsx`):**
   * Modify [StreamPlayer.tsx](file:///Users/zulqarnain/GitProjects/bcss/gmr-poc/gmr-frontend/src/features/live/components/StreamPlayer.tsx).
   * Add a transparent SVG/Canvas element overlaid absolute over the `<video>`.
   * Draw boxes using relative CSS layout parameters (`left`, `top`, `width`, `height` in %).
   * Expose a toggling UI button to turn bounding boxes ON or OFF.
