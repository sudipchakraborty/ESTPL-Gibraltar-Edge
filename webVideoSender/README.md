# MediaMTX outbound publisher

This optional module publishes the latest annotated OpenCV frame to MediaMTX
without replacing or changing the existing local `videoSender`.

## Local test

1. Run MediaMTX locally with its RTSP and WebRTC listeners enabled.
2. Set `media_publisher.enabled` to `true` in the Jutemill `config.json`.
3. Start `Projects/Jutemill/main.py`.
4. Build the frontend with:

```env
VITE_FACTORY_STREAM_MODE=mediamtx
VITE_FACTORY_STREAM_URL=http://127.0.0.1:8889/jutemill-gate/whep
VITE_FACTORY_ALERT_URL=http://127.0.0.1:5000
```

The home/VPS deployment only changes the publish and WHEP URLs. The processing
loop and existing local WebRTC sender remain available.
