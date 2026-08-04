"""Run an independent VideoSender with a generated test pattern."""

from __future__ import annotations

import argparse
import signal
import time

import cv2
import numpy as np

from .sender import VideoSender


def main() -> None:
    parser = argparse.ArgumentParser(description="VisualAI WebRTC video sender")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    running = True

    def stop(*_: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    sender = VideoSender(host=args.host, port=args.port, fps=args.fps)
    sender.start()
    started_at = time.monotonic()
    try:
        while running:
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            elapsed = time.monotonic() - started_at
            x = int((elapsed * 120) % 600)
            cv2.rectangle(frame, (x, 140), (x + 40, 180), (0, 200, 255), -1)
            cv2.putText(
                frame,
                "VisualAI VideoSender test",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            sender.publish(frame)
            time.sleep(1.0 / args.fps)
    finally:
        sender.close()


if __name__ == "__main__":
    main()
