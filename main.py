"""Thin launcher for the Gibraltar Edge application.

Run from the repository root with::

    python main.py
"""

from __future__ import annotations

import multiprocessing as mp

from Projects.main import main


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
