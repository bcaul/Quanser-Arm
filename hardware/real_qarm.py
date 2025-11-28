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
from typing import List

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

    def __init__(self) -> None:
        # Placeholder for the Quanser hardware connection. Keep this import-free
        # so the package is importable without the vendor SDK present.
        self._current_joint_positions = [0.0, 0.0, 0.0, 0.0]

        self.exitStack = ExitStack()
        register(self.exitStack.close)

        with suppress_stdout():
            self.arm = self.exitStack.enter_context(QArm(hardware=1))
        self.home()

    def home(self) -> None:
        self.set_joint_positions([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.set_gripper_position(0.0)

    def set_joint_positions(self, joint_angles: List[float]) -> None:
        self._current_joint_positions = joint_angles

        self.arm.read_write_std(phiCMD=joint_angles, baseLED=np.array([0, 1, 0], dtype=np.float64))

    def get_joint_positions(self) -> list[float]:
        return self._current_joint_positions

    def set_gripper_position(self, closure: float) -> None:
        self.arm.read_write_std(gprCMD=closure, baseLED=np.array([0, 1, 0], dtype=np.float64))

    def set_gripper_positions(self, angles: List[float]) -> None:
        self._not_ready("set_gripper_positions")

    def open_gripper(self) -> None:
        self.set_gripper_position(0.0)

    def close_gripper(self) -> None:
        self.set_gripper_position(0.55)

    def _not_ready(self, call: str) -> None:
        raise NotImplementedError(
            f"RealQArm.{call} requires the Quanser hardware SDK and a connected robot. "
            "Use mode='sim' during the hackathon, or supply a hardware client when available."
        )
