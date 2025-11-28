"""
Hardware-based QArm controller that will wrap the Quanser Python API.

Quanser-specific imports and calls are intentionally omitted; fill them in when
the hardware SDK is available in the hackathon environment.
"""

from __future__ import annotations

import numpy as np
from atexit import register
from common.qarm_base import QArmBase
from contextlib import contextmanager, redirect_stdout
from contextlib import ExitStack
from os import devnull
from pal.products.qarm import QArm
from typing import List, Sequence

@contextmanager
def suppress_stdout():
    with open(devnull, "w") as fnull:
        with redirect_stdout(fnull):
            yield

class RealQArm(QArmBase):
    """
    Hardware-based QArm controller using the Quanser Python API.

    This class will wrap the vendor-provided SDK to control the physical QArm.
    For now, it only defines the shape of the interface and where future code
    should be added.
    """

    def __init__(self, client: QArm | None = None) -> None:
        # Default to the vendor SDK QArm if no client is provided (keeps the
        # import path stable while letting tests inject a stub).
        self._current_joint_positions = [0.0, 0.0, 0.0, 0.0]

        self.exitStack = ExitStack()
        register(self.exitStack.close)

        if client is not None:
            self.arm = client
        else:
            with suppress_stdout():
                self.arm = self.exitStack.enter_context(QArm(hardware=1))
        self.home()

    def home(self) -> None:
        print("[RealQArm] home()")
        self.set_joint_positions([0.0, 0.0, 0.0, 0.0])
        self.set_gripper_position(0.0)

    def set_joint_positions(self, joint_angles: Sequence[float]) -> None:
        print(f"[RealQArm] set_joint_positions({joint_angles})")
        # Only the first four joints are driven; clamp excess input to match.
        phi = list(joint_angles)[:4]
        if len(phi) < 4:
            phi = phi + [0.0] * (4 - len(phi))
        self._current_joint_positions = phi
        self.arm.read_write_std(phiCMD=np.array(phi, dtype=np.float64), baseLED=np.array([0, 1, 0], dtype=np.float64))

    def get_joint_positions(self) -> list[float]:
        print("[RealQArm] get_joint_positions() ->", self._current_joint_positions)
        return self._current_joint_positions

    def set_gripper_position(self, closure: float) -> None:
        print(f"[RealQArm] set_gripper_position({closure})")
        self.arm.read_write_std(gprCMD=closure, baseLED=np.array([0, 1, 0], dtype=np.float64))

    def set_gripper_positions(self, angles: Sequence[float] | float) -> None:
        print(f"[RealQArm] set_gripper_positions({angles})")
        # Mirror SimQArm: accept a scalar or a sequence and forward the first value.
        if isinstance(angles, (int, float)):
            self.set_gripper_position(float(angles))
            return
        if not angles:
            return
        self.set_gripper_position(float(angles[0]))

    def open_gripper(self) -> None:
        print("[RealQArm] open_gripper()")
        self.set_gripper_position(0.0)

    def close_gripper(self) -> None:
        print("[RealQArm] close_gripper()")
        self.set_gripper_position(0.55)

    def _not_ready(self, call: str) -> None:
        raise NotImplementedError(
            f"RealQArm.{call} requires the Quanser hardware SDK and a connected robot. "
            "Use mode='sim' during the hackathon, or supply a hardware client when available."
        )
