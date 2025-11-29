"""
Scripted pick-and-place in plain Python.

What it shows:
- A small dictionary of poses you can edit without hunting through code.
- A simple sequence that opens/closes the gripper between waypoints.
- Prints every move so you see which joint set is being sent.

Run it (opens Panda3D viewer by default):
    python -m demos.pick_and_place
Tweak `POSES` and `SEQUENCE` below to match where your hoop and stand sit.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from pathlib import Path
import math
import json

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase

MODE = "sim"
USE_PANDA_VIEWER = True
USE_PYBULLET_GUI = False
STEP_DELAY_S = 1.0

# Joint targets in `(yaw, shoulder, elbow, wrist)` order.
POSES: dict[str, tuple[float, float, float, float]] = {
    "home": (0.0, 0.55, 0.0, 0.0),
    "above_pick": (0.25, 0.35, -0.1, 0.0),
    "at_pick": (0.25, 0.6, -0.35, 0.0),
    "lift": (0.25, 0.35, -0.1, 0.0),
    "above_place": (-0.75, 0.45, 0.05, 0.2),
    "at_place": (-0.75, 0.65, -0.25, 0.2),
}

SEQUENCE: list[tuple[str, str]] = [
    ("Move above the hoop", "above_pick"),
    ("Lower to grab height", "at_pick"),
    ("Close gripper", "close"),
    ("Lift the hoop", "lift"),
    ("Slide over to the stand", "above_place"),
    ("Lower onto the stand", "at_place"),
    ("Open gripper", "open"),
    ("Return home", "home"),
]

GRIPPER_OPEN_ANGLE = 0.0
GRIPPER_CLOSED_ANGLE = 0.55


def go_to(arm: QArmBase, pose_name: str, wait: float) -> None:
    pose = POSES[pose_name]
    print(f"[PickPlace] {pose_name}: {pose}")
    arm.set_joint_positions(pose)
    time.sleep(wait)


def run_sequence(arm: QArmBase, *, repeats: int | None = None) -> None:
    print("[PickPlace] Joint order:", ", ".join(DEFAULT_JOINT_ORDER))
    count = 0
    while repeats is None or count < repeats:
        for action, target in SEQUENCE:
            if target == "open":
                print("[PickPlace] Opening gripper")
                arm.set_gripper_position(GRIPPER_OPEN_ANGLE)
            elif target == "close":
                print("[PickPlace] Closing gripper")
                arm.set_gripper_position(GRIPPER_CLOSED_ANGLE)
            else:
                go_to(arm, target, STEP_DELAY_S)
            time.sleep(STEP_DELAY_S)
        count += 1


def main() -> None:
    auto_step = not USE_PANDA_VIEWER
    mode = MODE.lower()
    mirror_mode = mode == "mirror"
    effective_mode = "hardware" if mirror_mode else mode
    arm = make_qarm(
        mode=effective_mode,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        auto_step=auto_step,
        mirror_sim=mirror_mode,
    )
    mirror_status = "on" if mirror_mode else "off"
    print(f"[PickPlace] Connected (mode={mode}, mirror={mirror_status}); moving to home.")
    arm.home()
    time.sleep(STEP_DELAY_S)

    # Spawn demo hoops before the viewer starts so the viewer instantiates
    # their Panda3D nodes during setup (match demos/hoop_segments behavior).
    try:
        env = getattr(arm, "env", None)
        # Skip if something already spawned kinematic objects.
        if env is None or not getattr(env, "kinematic_objects", None):
            try:
                import demos.hoop_segments as _hoop_demo

                try:
                    _hoop_demo._place_first_green_over_accent(arm)
                except Exception:
                    pass
                try:
                    _hoop_demo.add_hoops(arm, _hoop_demo.DEFAULT_HOOP_DEFS)
                    print("[PickPlace] Spawned demo hoops from demos.hoop_segments.")
                except Exception as e:
                    print(f"[PickPlace] Failed to spawn demo hoops: {e}")
            except Exception:
                # No hoop demo available; continue silently.
                pass
        else:
            print("[PickPlace] Kinematic objects already present; skipping demo hoop spawn.")
    except Exception:
        pass

    try:
        if USE_PANDA_VIEWER:
            import threading
            from sim.panda_viewer import PandaArmViewer, PhysicsBridge

            def launch_viewer() -> None:
                env = getattr(arm, "env", None)
                if env is None:
                    print("[PickPlace] Viewer unavailable (no sim env).")
                    return
                args = SimpleNamespace(
                    time_step=env.time_step,
                    hide_base=False,
                    hide_accents=False,
                    probe_base_collision=False,
                    show_sliders=False,
                    reload_meshes=False,
                )
                bridge = PhysicsBridge(time_step=env.time_step, env=env, reset=False)
                PandaArmViewer(bridge, args).run()

            # Attempt a single automatic pick of the first hoop defined in
            # `demos/hoop_positions.json` (or the demo defaults) before the
            # motion thread so the action is visible in the viewer.
            try:
                _auto_pick_first_hoop(arm)
            except Exception:
                pass

            motion = threading.Thread(target=run_sequence, args=(arm,), kwargs={"repeats": None}, daemon=True)
            motion.start()
            launch_viewer()  # blocks until window closes
        else:
            run_sequence(arm, repeats=None)
    except KeyboardInterrupt:
        print("\n[PickPlace] Sequence interrupted.")
    finally:
        try:
            print("[PickPlace] Returning to home...")
            arm.home()
        except Exception:
            pass


def _auto_pick_first_hoop(arm: QArmBase) -> bool:
    """Pick the first enabled hoop from `demos/hoop_positions.json` using
    an analytic yaw and PyBullet IK for the remaining joints.

    Mathematical derivation (brief):
    - Let the hoop target be p = (x, y, z) in robot base frame. The base
      yaw rotation about Z that points the arm toward p is yaw = atan2(y, x).
    - After applying yaw, the remaining arm is approximately planar. For a
      planar 2R arm (shoulder/elbow) the standard law-of-cosines solution is
      used to compute shoulder/elbow angles that reach a wrist target. Here
      we rely on PyBullet's `calculateInverseKinematics` to solve the full
      chain numerically, but we compute yaw analytically and enforce it so
      the base rotation is deterministic.

    Returns True if the routine executed without fatal errors.
    """
    try:
        import pybullet as p
    except Exception:
        print("[PickPlace] PyBullet not available; cannot auto-pick.")
        return False

    env = getattr(arm, "env", None)
    if env is None:
        print("[PickPlace] No simulation env; cannot auto-pick.")
        return False

    # Load hoop positions (fallback to demo defaults if file missing)
    demo_pos_file = Path(__file__).resolve().parent / "hoop_positions.json"
    defs = None
    if demo_pos_file.exists():
        try:
            with demo_pos_file.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list) and loaded:
                defs = [d for d in loaded if d.get("enabled", True)]
        except Exception:
            defs = None

    # Fallback: try to import the demo defaults
    if not defs:
        try:
            import demos.hoop_segments as _hd

            defs = [_d for _d in getattr(_hd, "DEFAULT_HOOP_DEFS", []) if _d.get("enabled", True)]
        except Exception:
            defs = []

    if not defs:
        print("[PickPlace] No hoop definitions found; skipping auto-pick.")
        return False

    first = defs[0]
    pos = tuple(first.get("position", (0.0, -0.3, 0.08)))
    tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

    # Analytic yaw to face the hoop
    yaw = math.atan2(ty, tx)

    # Choose a small upward offset for approach and a small lower offset for grasp
    approach_z = tz + 0.08
    grasp_z = tz + 0.01

    approach = (tx, ty, approach_z)
    grasp = (tx, ty, grasp_z)

    # Compute an IK solution for an approach pose (PyBullet provides numeric IK)
    try:
        ee_link_idx = getattr(env, "_gripper_base_link_index", None)
        if ee_link_idx is None:
            # fall back to END-EFFECTOR link index if present in mapping
            idx_map = getattr(env, "link_name_to_index", {})
            ee_link_idx = idx_map.get("END-EFFECTOR") or idx_map.get("GRIPPER_BASE")
        if ee_link_idx is None:
            print("[PickPlace] Could not determine end-effector link for IK.")
            return False

        sol = p.calculateInverseKinematics(env.robot_id, ee_link_idx, approach, physicsClientId=env.client)
    except Exception as e:
        print(f"[PickPlace] IK solve failed: {e}")
        return False

    # Map solution to our arm ordering (joint indices for the arm DOFs)
    try:
        joint_indices = getattr(arm, "joint_order", None)
        if joint_indices is None:
            joint_indices = getattr(env, "movable_joint_indices", [])
        targets = [float(sol[j]) for j in joint_indices]
    except Exception:
        print("[PickPlace] Failed to map IK solution to joint targets.")
        return False

    # Enforce analytic yaw as the first joint target (safe, intuitive)
    if targets:
        targets[0] = yaw

    # Move to approach, then lower, close gripper, and lift.
    try:
        arm.set_joint_positions(targets)
        time.sleep(0.6)

        # IK for grasp (lower)
        sol2 = p.calculateInverseKinematics(env.robot_id, ee_link_idx, grasp, physicsClientId=env.client)
        targets2 = [float(sol2[j]) for j in joint_indices]
        targets2[0] = yaw
        arm.set_joint_positions(targets2)
        time.sleep(0.5)

        # Close gripper
        try:
            arm.set_gripper_position(GRIPPER_CLOSED_ANGLE)
        except Exception:
            pass
        time.sleep(0.5)

        # Lift back to approach
        arm.set_joint_positions(targets)
        time.sleep(0.6)
    except Exception as e:
        print(f"[PickPlace] Movement failed during auto-pick: {e}")
        return False

    print("[PickPlace] Auto-pick attempted (check viewer for result).")
    return True


if __name__ == "__main__":
    main()
