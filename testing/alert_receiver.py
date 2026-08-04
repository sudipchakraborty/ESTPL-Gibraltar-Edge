"""Compatibility launcher for the independent AlertServer package."""

import sys
from pathlib import Path


REPOSITORY_DIR = Path(__file__).resolve().parents[1]
if str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

from alertServer import AlertServer


if __name__ == "__main__":
    AlertServer(host="127.0.0.1", port=5000).run()
