"""Streaming module.

Modular, low-latency live preview of camera RTSP streams for the browser
(zone-binding UI). Pipeline:

    RTSP (camera)  --ffmpeg-->  MediaMTX  --WebRTC/WHEP (or HLS)-->  Browser

Design:
- `base.StreamPublisher`  : abstract publisher contract (swap ffmpeg for GStreamer, etc.)
- `mediamtx.MediaMTXManager` : builds MediaMTX path names and public WebRTC/HLS URLs.
- `ffmpeg_publisher.FFmpegPublisher` : republishes one camera's RTSP into MediaMTX.
- `manager.StreamManager` : process-wide singleton; ref-counts viewers, auto-stops idle streams.
- `snapshot` : grabs a single JPEG frame from RTSP for the polygon-drawing canvas.
"""
