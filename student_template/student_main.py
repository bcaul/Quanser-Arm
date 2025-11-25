"""
Minimal student-facing entry point: drive the QArm in simulation via joint commands.

Quick API reminder (common to sim + hardware):
- `get_joint_positions()` -> list[float] ordered (yaw, shoulder, elbow, wrist)
- `set_joint_positions(q)` sends that list/tuple to the robot
- `home()` moves to a safe default pose

The sim is meant to be viewed in Panda3D (`python -m sim.panda_viewer`). The PyBullet
GUI is optional and best used for debugging joint values; this script launches Panda3D
by default unless you opt out.

Run with:
    python -m student_template.student_main                      # opens Panda3D viewer
    python -m student_template.student_main --headless           # headless (no Panda3D window)
    python -m student_template.student_main --pybullet-gui       # enable PyBullet's debug GUI
"""

from __future__ import annotations

import argparse
import math
import threading
import time
from types import SimpleNamespace

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase


def demo_motion(arm: QArmBase, duration: float, step_s: float, stop_event: threading.Event | None = None) -> None:
    """
    Walk through a repeating set of waypoints in joint space.

    Each waypoint is a list/tuple in the documented order:
    (yaw, shoulder, elbow, wrist)

    The sequence loops until `duration` elapses (<=0 means run forever) or a stop event is set.
    """

    # 1) Waypoints: define explicit joint targets in the documented order.
    #    You can add as many as you like; the API just takes a sequence of floats.
    waypoints = [
        ("Home (from get_joint_positions)", arm.get_joint_positions()),
        ("Stretch forward", [0.0, -0.5, 0.7, 0.0]),
        ("Lift up", [0.0, -0.2, 0.3, 0.0]),
        ("Rotate wrist", [0.0, -0.2, 0.3, 0.8]),
    ]

    start = time.time()
    end_time = None if duration <= 0 else start + duration
    base_pose = arm.get_joint_positions()

    def time_left() -> bool:
        return end_time is None or time.time() < end_time

    while time_left():
        # Step through waypoints.
        for name, pose in waypoints:
            if stop_event is not None and stop_event.is_set():
                return
            if not time_left():
                return
            print(f"[Demo] Moving to waypoint: {name}  ({DEFAULT_JOINT_ORDER})")
            arm.set_joint_positions(pose)
            time.sleep(1.0)
            # brief dwell between waypoints
            time.sleep(step_s)


def main() -> None:
    # -------- CLI and setup --------
    parser = argparse.ArgumentParser(description="Simple joint-space driving loop for the QArm.")
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="How long to run the demo (seconds). Use 0 or negative to run indefinitely.",
    )
    parser.add_argument("--step", type=float, default=0.02, help="Sleep between joint commands (seconds).")
    parser.add_argument(
        "--pybullet-gui",
        action="store_true",
        help="Open PyBullet's debug GUI (Panda3D viewport is the recommended renderer).",
    )
    parser.add_argument(
        "--headless",
        dest="panda_viewer",
        action="store_false",
        default=True,
        help="Run without launching the Panda3D viewer window.",
    )
    args = parser.parse_args()

    # Use manual stepping; when Panda viewer is active it drives the simulation clock.
    real_time = False
    auto_step = not args.panda_viewer
    arm = make_qarm(mode="sim", gui=args.pybullet_gui, real_time=real_time, auto_step=auto_step)
    arm.home()
    time.sleep(0.1)

    # -------- Viewer wiring (optional) --------
    stop_event = threading.Event()
    motion_thread: threading.Thread | None = None

    def launch_viewer() -> None:
        from sim.panda_viewer import PandaArmViewer, PhysicsBridge

        viewer_args = SimpleNamespace(
            time_step=arm.env.time_step,
            base_mesh=None,
            base_collision_mesh=None,
            base_mesh_scale=0.001,
            base_yaw=180.0,
            base_friction=0.8,
            base_restitution=0.0,
            green_accent=None,
            blue_accent=None,
            hide_base=False,
            hide_accents=False,
            probe_base_collision=False,
        )
        physics = PhysicsBridge(
            time_step=arm.env.time_step,
            base_mesh=None,
            base_collision_mesh=None,
            base_mesh_scale=0.001,
            base_yaw_deg=180.0,
            base_friction=0.8,
            base_restitution=0.0,
            env=getattr(arm, "env", None),
            reset=False,
        )
        app = PandaArmViewer(physics, viewer_args)
        app.run()

    try:
        # -------- Motion loop --------
        if args.panda_viewer:
            motion_thread = threading.Thread(
                target=demo_motion, args=(arm, args.duration, args.step, stop_event), daemon=True
            )
            motion_thread.start()
            launch_viewer()
            stop_event.set()
        else:
            demo_motion(arm, duration=args.duration, step_s=args.step, stop_event=stop_event)
    except KeyboardInterrupt:
        print("\nStopping demo...")
        stop_event.set()
    finally:
        if motion_thread is not None and motion_thread.is_alive():
            motion_thread.join(timeout=1.0)
        # Park the arm back at home for a clean exit.
        try:
            arm.home()
        except Exception:
            pass


if __name__ == "__main__":
    main()
