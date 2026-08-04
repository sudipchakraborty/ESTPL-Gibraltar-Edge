# VideoSender

Reusable latest-frame WebRTC publisher for OpenCV applications.

## Application API

```python
from videoSender import VideoSender

video_sender = VideoSender(host="0.0.0.0", port=8000, fps=10)
video_sender.start()

while True:
    frame = get_processed_opencv_frame()
    video_sender.publish(frame)
```

Open `http://EDGE_IP:8000` in a browser and select **Connect live video**.

`publish()` only replaces the latest buffered frame. It does not wait for the
network or queue old frames, so a slow viewer does not delay camera processing.

## Standalone test

```powershell
python -m videoSender --host 0.0.0.0 --port 8000
```

This publishes a generated moving test pattern.

## Remote networks

Direct remote viewing requires HTTPS signaling and ICE configuration. Provide
STUN/TURN servers through the `ice_servers` constructor argument:

```python
ice_servers=[
    {"urls": "stun:stun.example.com:3478"},
    {
        "urls": "turn:turn.example.com:3478",
        "username": "visualai",
        "credential": "secret",
    },
]
```

For many viewers, publish one uplink to a central WebRTC SFU instead of creating
one peer connection and one edge upload per browser.
