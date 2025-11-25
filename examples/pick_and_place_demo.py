"""
Example script showing how a higher-level strategy might interact with the QArm API.

This keeps the motion logic minimal; students can import `run_demo` and swap in
their own kinematics or planning code while reusing the same QArmBase interface.
"""

from __future__ import annotations

import math
import time

from api.factory import make_qarm
from common.qarm_base import QArmBase


def run_demo(arm: QArmBase, duration: float = 6.0, step_s: float = 0.02) -> None:
    """
    Simple placeholder flow:

    - Home the arm
    - Move joint 0 through a small sinusoid while holding other joints steady
    """
    arm.home()
    time.sleep(0.1)
    base = arm.get_joint_positions()
    start = time.time()
    while time.time() - start < duration:
        t = time.time() - start
        q = list(base)
        if q:
            q[0] = 0.35 * math.sin(t)
        arm.set_joint_positions(q)
        time.sleep(step_s)


if __name__ == "__main__":
    arm = make_qarm(mode="sim")
    try:
        run_demo(arm)
    except KeyboardInterrupt:
        print("\nStopping demo...")
