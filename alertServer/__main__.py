"""Run AlertServer as an independent process."""

from __future__ import annotations

import argparse
import os

from .server import AlertServer


def main() -> None:
    parser = argparse.ArgumentParser(description="VisualAI alert relay server")
    parser.add_argument(
        "--host",
        default=os.getenv("ALERT_SERVER_HOST", "0.0.0.0"),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ALERT_SERVER_PORT", "5000")),
    )
    args = parser.parse_args()
    AlertServer(host=args.host, port=args.port).run()


if __name__ == "__main__":
    main()
