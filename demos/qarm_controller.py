"""QArm kinematics controller utilities.

Provides forward and inverse kinematics wrappers and simple pick/place
sequence helpers that use the QArm utilities present in the repository.

Functions:
  - load_hoop_positions(json_path)
  - forward_kinematics(theta1, theta2, theta3, theta4)
  - inverse_kinematics(x, y, z, elbow='best')
  - move_to(arm, x, y, z, yaw=0.0)
  - pick_hoop(arm, x, y, z)
  - place_hoop(arm, px, py, pz)
  - sort_all_hoops(arm, hoops, pole_pos)

This module is intentionally small and uses the existing
`hardware.hal.products.qarm.QArmUtilities` implementation for the math.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    from hardware.hal.products.qarm import QArmUtilities
except Exception:
    QArmUtilities = None


def load_hoop_positions(json_path: str | Path) -> list[tuple[float, float, float]]:
    """Load hoop coordinates from a JSON file.

    Expected JSON format: an array of objects with a `position` array or
    direct objects like {"x":..., "y":..., "z":...}.
    Returns a list of (x,y,z) tuples.
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(p)
    data = json.loads(p.read_text())
    hoops: list[tuple[float, float, float]] = []
    for entry in data:
        if isinstance(entry, dict):
            if "position" in entry:
                pos = entry["position"]
                hoops.append((float(pos[0]), float(pos[1]), float(pos[2])))
            elif {"x", "y", "z"}.issubset(entry.keys()):
                hoops.append((float(entry["x"]), float(entry["y"]), float(entry["z"])))
    return hoops


def forward_kinematics(theta1: float, theta2: float, theta3: float, theta4: float) -> Tuple[float, float, float]:
    """Compute end-effector position using QArm parameters.

    Returns (x, y, z). This delegates to `QArmUtilities.forward_kinematics`.
    Angles are in radians.
    """
    if QArmUtilities is None:
        raise RuntimeError("QArmUtilities is unavailable in this environment")
    util = QArmUtilities()
    phi = [float(theta1), float(theta2), float(theta3), float(theta4)]
    p4, _ = util.forward_kinematics(phi)
    return float(p4[0]), float(p4[1]), float(p4[2])


def inverse_kinematics(x: float, y: float, z: float, elbow: str = "best") -> Tuple[float, float, float, float]:
    """Compute joint angles (theta1, theta2, theta3, theta4) for target (x,y,z).

    elbow: 'up', 'down', or 'best' (closest to 0 initial guess). Returns a 4-tuple phi angles in radians.
    Uses QArmUtilities.inverse_kinematics; if unavailable, raises.
    """
    if QArmUtilities is None:
        raise RuntimeError("QArmUtilities is unavailable in this environment")
    util = QArmUtilities()
    # p is expected as [x, y, z]
    p = [float(x), float(y), float(z)]
    # gamma (wrist rotation) set to zero by default
    gamma = 0.0
    # phi_prev initialize at zeros to pick a reasonable branch
    phi_prev = [0.0, 0.0, 0.0, 0.0]
    phi_all, phi_opt = util.inverse_kinematics(p, gamma, phi_prev)
    # phi_all is 4x4 matrix, phi_opt is optimal
    if elbow == "up":
        # pick one of the solutions: try columns 0/1/2/3 for elbow-up heuristic
        sol = phi_all[:, 0]
    elif elbow == "down":
        sol = phi_all[:, 1]
    else:
        sol = phi_opt
    return float(sol[0]), float(sol[1]), float(sol[2]), float(sol[3])


def move_to(arm: Any, x: float, y: float, z: float, yaw: float = 0.0, *, verbose: bool = False) -> bool:
    """Compute IK and command the arm to the target pose.

    This function sends joint angles to `arm.set_joint_positions`.
    Returns True on success.
    """
    # First, try analytic/geometric IK via QArmUtilities (if available)
    try:
        theta1, theta2, theta3, theta4 = inverse_kinematics(x, y, z)
        targets = [theta1, theta2, theta3, theta4]
        # If caller provided an explicit yaw, enforce it on joint 0
        if yaw is not None and float(yaw) != 0.0:
            targets[0] = float(yaw)
        arm.set_joint_positions(targets)
        if verbose:
            print(f"[move_to] Sent joint targets (analytic IK): {targets}")
        return True
    except Exception as analytic_err:
        if verbose:
            print(f"[move_to] Analytic IK failed or unavailable: {analytic_err}")

    # Fallback: use PyBullet numeric IK from the arm's environment (if present)
    env = getattr(arm, "env", None)
    try:
        import pybullet as p
    except Exception:
        p = None

    if env is None or p is None:
        if verbose:
            print("[move_to] No analytic IK and no PyBullet environment available for numeric IK.")
        return False

    # Locate an appropriate IK link index
    link_idx = getattr(env, "_gripper_base_link_index", None)
    if link_idx is None:
        idx_map = getattr(env, "link_name_to_index", {})
        link_idx = idx_map.get("GRIPPER_BASE") or idx_map.get("END-EFFECTOR")
    if link_idx is None:
        if verbose:
            print("[move_to] Could not find gripper/EE link index for numeric IK.")
        return False

    try:
        ik_solution = p.calculateInverseKinematics(env.robot_id, link_idx, (float(x), float(y), float(z)), physicsClientId=env.client)
        if not ik_solution:
            if verbose:
                print(f"[move_to] PyBullet IK returned no solution for ({x},{y},{z}).")
            return False
        joint_indices = getattr(arm, "joint_order", None)
        if joint_indices is None:
            joint_indices = getattr(env, "movable_joint_indices", [])
        targets = [float(ik_solution[j]) for j in joint_indices]
        # enforce base yaw analytically
        if targets:
            targets[0] = math.atan2(y, x)
        arm.set_joint_positions(targets)
        if verbose:
            print(f"[move_to] Sent joint targets (numeric IK): {targets}")
        return True
    except Exception as e:
        if verbose:
            print(f"[move_to] Numeric IK or command failed: {e}")
        return False


def slow_vertical_move(
    arm: Any,
    x: float,
    y: float,
    z_from: float,
    z_to: float,
    *,
    steps: int = 6,
    delay: float = 0.12,
    yaw: float | None = None,
    verbose: bool = False,
) -> bool:
    """Interpolate vertical motion in small steps to avoid diagonal plunges.

    Keeps X/Y fixed and calls `move_to` for each intermediate Z.
    """
    if steps < 1:
        steps = 1
    target_yaw = yaw if yaw is not None else math.atan2(y, x)
    for i in range(1, steps + 1):
        alpha = i / float(steps)
        z_i = float(z_from) + (float(z_to) - float(z_from)) * alpha
        ok = move_to(arm, x, y, z_i, yaw=target_yaw, verbose=verbose)
        if not ok:
            if verbose:
                print(f"[slow_vertical_move] IK failed at intermediate z={z_i:.3f}")
            return False
        time.sleep(delay)
    return True


def pick_hoop(
    arm: Any,
    x: float,
    y: float,
    z: float,
    *,
    hover: float = 0.10,
    gripper_close: float = 0.55,
    move_delay_s: float = 0.6,
    gripper_settle_s: float = 0.6,
    max_attempts: int = 3,
    nudge_xy: float = 0.004,
    verbose: bool = True,
) -> bool:
    """Robust pick sequence using slow vertical moves and small XY nudges.

    Tries several nudges and small extra descent offsets to reliably capture hoops.
    """
    env = getattr(arm, "env", None)

    # Ensure gripper is open
    try:
        arm.set_gripper_position(0.0)
    except Exception:
        pass

    nudges = [(0.0, 0.0), (nudge_xy, 0.0), (-nudge_xy, 0.0), (0.0, nudge_xy), (0.0, -nudge_xy)]
    z_offsets = [0.0, -0.002, -0.004]

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"[pick_hoop] Attempt {attempt}/{max_attempts} for hoop at ({x:.3f},{y:.3f},{z:.3f})")

        for dx, dy in nudges:
            tx, ty = x + dx, y + dy
            target_yaw = math.atan2(ty, tx)

            # Move to hover at the nudged xy
            if not slow_vertical_move(arm, tx, ty, z + hover + 0.02, z + hover, steps=4, delay=move_delay_s / 3.0, yaw=target_yaw, verbose=verbose):
                continue
            time.sleep(move_delay_s)

            for z_off in z_offsets:
                grasp_z = z + z_off
                # Slow, vertical descent from hover to grasp_z
                if not slow_vertical_move(arm, tx, ty, z + hover, grasp_z, steps=6, delay=move_delay_s / 3.0, yaw=target_yaw, verbose=verbose):
                    continue
                time.sleep(move_delay_s / 2.0)

                # Close gripper
                try:
                    arm.set_gripper_position(gripper_close)
                    if verbose:
                        print(f"[pick_hoop] Gripper closing (angle={gripper_close:.3f})")
                except Exception as e:
                    if verbose:
                        print(f"[pick_hoop] Gripper close failed: {e}")
                    continue
                time.sleep(gripper_settle_s)

                # Lift back to hover slowly
                if not slow_vertical_move(arm, tx, ty, grasp_z, z + hover, steps=6, delay=move_delay_s / 3.0, yaw=target_yaw, verbose=verbose):
                    continue
                time.sleep(move_delay_s / 2.0)

                # Check grasp
                grasp_ok = False
                try:
                    if env is not None and hasattr(env, "get_active_grasp_info"):
                        info = env.get_active_grasp_info()
                        grasp_ok = info is not None
                        if verbose:
                            print(f"[pick_hoop] Grasp info: {info}")
                except Exception:
                    grasp_ok = False

                if grasp_ok:
                    if verbose:
                        print(f"[pick_hoop] Successfully picked hoop at ({x:.3f}, {y:.3f}, {z:.3f})")
                    return True

                # Open and try next nudge
                try:
                    arm.set_gripper_position(0.0)
                except Exception:
                    pass
                time.sleep(0.12)

    if verbose:
        print(f"[pick_hoop] Failed to pick hoop at ({x:.3f}, {y:.3f}, {z:.3f}) after {max_attempts} attempts")
    return False


def place_hoop(
    arm: Any,
    px: float,
    py: float,
    pz: float,
    *,
    hover: float = 0.10,
    gripper_open: float = 0.0,
    transit_z: float | None = None,
    slow_descent: bool = True,
    move_delay_s: float = 0.6,
    verbose: bool = True,
) -> bool:
    """Place sequence: raise to transit height, move horizontally, slow descend to release, open, and retreat.
    """
    # Transit height: prefer provided or compute conservatively above pole
    tx_z = transit_z if transit_z is not None else max(0.20, pz + 0.08)

    # If not already at transit, raise vertically from current approximate z (hover) to transit_z
    # We'll call slow_vertical_move from current hover to transit.
    # Move horizontally to pole at transit height
    if verbose:
        print(f"[place_hoop] Moving to transit height {tx_z:.3f} and then to pole XY ({px:.3f},{py:.3f})")
    if not move_to(arm, px, py, tx_z, verbose=verbose):
        # try numeric IK move if needed
        if not move_to(arm, px, py, tx_z, verbose=verbose):
            return False
    time.sleep(move_delay_s / 2.0)

    # Slow descent from transit_z to release_z
    release_z = pz + 0.01
    if slow_descent:
        if not slow_vertical_move(arm, px, py, tx_z, release_z, steps=8, delay=move_delay_s / 4.0, yaw=math.atan2(py, px), verbose=verbose):
            return False
    else:
        if not move_to(arm, px, py, release_z, verbose=verbose):
            return False
    time.sleep(move_delay_s / 2.0)

    # Open gripper to drop while hovering above the pole
    try:
        arm.set_gripper_position(gripper_open)
        if verbose:
            print("[place_hoop] Gripper opened (drop).")
    except Exception as e:
        if verbose:
            print(f"[place_hoop] Gripper open failed: {e}")
        return False
    time.sleep(move_delay_s / 2.0)

    # Retreat back to transit_z
    if not slow_vertical_move(arm, px, py, release_z, tx_z, steps=6, delay=move_delay_s / 4.0, yaw=math.atan2(py, px), verbose=verbose):
        return False
    time.sleep(move_delay_s / 2.0)
    return True


def sort_all_hoops(arm: Any, hoops: Iterable[Tuple[float, float, float]], pole_pos: Tuple[float, float, float], *, verbose: bool = True) -> int:
    """Sort all hoops: pick each hoop and place it on the pole.

    Returns number of successfully placed hoops.
    """
    placed = 0
    # Start from home position if available
    try:
        if hasattr(arm, "home"):
            arm.home()
            time.sleep(0.6)
    except Exception:
        pass

    for i, (x, y, z) in enumerate(hoops):
        if verbose:
            print(f"[sort_all_hoops] Hoop {i+1}/{len(list(hoops))}: {x:.3f},{y:.3f},{z:.3f}")
        ok = pick_hoop(arm, x, y, z, verbose=verbose)
        if not ok:
            if verbose:
                print(f"[sort_all_hoops] Failed to pick hoop {i}")
            continue
        # Move to pole hover and place
        px, py, pz = pole_pos
        ok = place_hoop(arm, px, py, pz, verbose=verbose)
        if ok:
            placed += 1
        # Return home between cycles
        try:
            if hasattr(arm, "home"):
                arm.home()
                time.sleep(0.6)
        except Exception:
            pass

    return placed


if __name__ == "__main__":
    # Basic runner when executed directly (simulation expected)
    import argparse
    from api.factory import make_qarm

    parser = argparse.ArgumentParser(description="Run the QArm hoop sorter demo")
    parser.add_argument("--gui", action="store_true", help="Enable PyBullet GUI (show simulation window)")
    parser.add_argument("--real-time", action="store_true", help="Run in real-time mode (no auto-stepping)")
    parser.add_argument("--pole", nargs=3, type=float, default=(0.0, -0.35, 0.12), help="Pole position as three floats: X Y Z")
    args = parser.parse_args()

    # When real_time is enabled, don't auto-step; otherwise auto_step=True so the demo progresses.
    arm = make_qarm(mode="sim", gui=bool(args.gui), real_time=bool(args.real_time), auto_step=(not args.real_time))
    try:
        arm.home()
    except Exception:
        pass

    demo_json = Path(__file__).resolve().parent / "hoop_positions.json"
    hoops = load_hoop_positions(demo_json)
    pole = (float(args.pole[0]), float(args.pole[1]), float(args.pole[2]))
    print(f"Loaded {len(hoops)} hoops; starting sort to pole {pole}")
    n = sort_all_hoops(arm, hoops, pole, verbose=True)
    print(f"Done: placed {n}/{len(hoops)} hoops")
