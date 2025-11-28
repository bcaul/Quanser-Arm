"""
Small shared helpers for the student demos.
"""

from __future__ import annotations

import threading
from typing import Callable


def run_with_viewer(viewer_fn: Callable[[], None], worker_fn: Callable[[], None]) -> None:
    """
    Run `viewer_fn` on the main thread while `worker_fn` runs in a daemon thread.

    Needed for macOS/Panda3D where the window must be created on the main thread.
    """
    stop_event = threading.Event()

    def wrapped_worker() -> None:
        try:
            worker_fn()
        finally:
            stop_event.set()

    worker = threading.Thread(target=wrapped_worker, daemon=True)
    worker.start()
    try:
        viewer_fn()
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
