"""
Beginner-friendly quickstart for the QArm.

What it shows:
- Connects to the simulator (flip MODE to "hardware" when you have the real arm).
- Sends a few joint waypoints in the `(yaw, shoulder, elbow, wrist)` order.
- Opens and closes the gripper, then returns home.
- Optionally launches the Panda3D viewer so you can watch the motion.

Run it:
    python -m demos.student_main
Tweak the constants below to change timing, waypoints, or to turn the viewer/GUI on/off.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase

# ---- easy knobs to tweak ----
MODE = "sim"  # change to "hardware" once the real arm is wired up
USE_PANDA_VIEWER = True
USE_PYBULLET_GUI = False  # turn on for PyBullet debug sliders
TIME_STEP = 1.0 / 240.0
PAUSE_BETWEEN_MOVES = 1.0
SHOW_VIEWER_SLIDERS = False
RELOAD_MESHES = False

WAYPOINTS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("Reach forward", (0.0, 0.45, 0.25, 0.0)),
    ("Swing left", (-0.6, 0.55, 0.1, 0.2)),
    ("Swing right", (0.6, 0.45, -0.1, -0.2)),
]
GRIPPER_PAUSE_S = 1.0


def _print_joint_order() -> None:
    order = ", ".join(DEFAULT_JOINT_ORDER)
    print(f"[Student] Joint order: {order}")


def _run_viewer(arm: QArmBase) -> None:
    env = getattr(arm, "env", None)
    if env is None:
        print("[Viewer] Panda3D viewer needs the simulator; skipping.")
        return

    from sim.panda_viewer import PandaArmViewer, PhysicsBridge

    viewer_args = SimpleNamespace(
        time_step=env.time_step,
        hide_base=False,
        hide_accents=False,
        probe_base_collision=False,
        show_sliders=SHOW_VIEWER_SLIDERS,
        reload_meshes=RELOAD_MESHES,
    )
    bridge = PhysicsBridge(time_step=env.time_step, env=env, reset=False)
    PandaArmViewer(bridge, viewer_args).run()


def _play_motion(arm: QArmBase) -> None:
    _print_joint_order()
    print("[Student] Moving to home...")
    arm.home()
    time.sleep(PAUSE_BETWEEN_MOVES)

    for label, pose in WAYPOINTS:
        print(f"[Student] {label}: {pose}")
        arm.set_joint_positions(pose)
        time.sleep(PAUSE_BETWEEN_MOVES)

    print("[Student] Opening gripper")
    arm.open_gripper()
    time.sleep(GRIPPER_PAUSE_S)
    print("[Student] Closing gripper")
    arm.close_gripper()
    time.sleep(GRIPPER_PAUSE_S)


def main() -> None:
    auto_step = not USE_PANDA_VIEWER
    arm = make_qarm(
        mode=MODE,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        time_step=TIME_STEP,
        auto_step=auto_step,
    )
    print("[Student] Connected to QArm in", MODE, "mode.")

    try:
        if USE_PANDA_VIEWER:
            motion = threading.Thread(target=_play_motion, args=(arm,), daemon=True)
            motion.start()
            _run_viewer(arm)  # closes when you exit the viewer window
        else:
            _play_motion(arm)
    except KeyboardInterrupt:
        print("\n[Student] Stopping quickstart demo.")
    finally:
        try:
            print("[Student] Returning to home...")
            arm.home()
        except Exception:
            pass


if __name__ == "__main__":
    main()
