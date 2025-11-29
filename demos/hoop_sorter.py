"""
Hoop Sorting Module for Quanser QArm (4-DOF).

This module provides high-level routines to load hoop coordinates from a JSON file
and autonomously pick and place each hoop onto a pole using inverse kinematics
and gripper control.

Key functions:
  - load_hoop_coordinates(filepath): Load hoop positions from JSON.
  - move_to(arm, x, y, z, yaw=0): Cartesian motion using hybrid analytic yaw + PyBullet IK.
  - pick_hoop(arm, x, y, z): Move above, lower, grasp, and lift a hoop.
  - place_hoop(arm, px, py, pz): Move above pole, lower, release, retreat.
  - sort_all_hoops(arm, hoops, pole): Main orchestrator to place all hoops on pole.

Mathematical foundation (Inverse Kinematics):
  The QArm is a 4-DOF arm (yaw, shoulder, elbow, wrist) with the following
  kinematic chain:
    - Base (world) -> YAW (Z rotation) -> BICEP (shoulder, XY plane) -> 
      FOREARM (elbow, XY plane) -> END-EFFECTOR (wrist orientation)
    - Gripper assembly attaches to END-EFFECTOR (fixed relative to wrist).

  Approach:
    1. Analytic Yaw: yaw = atan2(target_y, target_x)
       This orients the base to face the target, making the arm approximately
       planar after the yaw rotation.
    2. Numeric IK: PyBullet's calculateInverseKinematics solves for the
       remaining 3 DOF (shoulder, elbow, wrist) given a gripper target position
       and orientation.

Usage Example:
    from api.factory import make_qarm
    from demos.hoop_sorter import load_hoop_coordinates, sort_all_hoops

    arm = make_qarm(mode="sim", gui=False, auto_step=True)
    arm.home()

    hoops = load_hoop_coordinates("demos/hoop_positions.json")
    pole_position = (0.5, -0.4, 0.12)  # (x, y, z)

    sort_all_hoops(arm, hoops, pole_position, verbose=True)
    arm.home()

Run standalone demo:
    python -m demos.hoop_sorter
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

try:
    import pybullet as p
except ImportError:
    p = None


def load_hoop_coordinates(filepath: str | Path) -> list[dict[str, float]]:
    """
    Load hoop coordinate positions from a JSON file.

    Expected JSON format:
        [
            {"position": [x1, y1, z1], ...},
            {"position": [x2, y2, z2], ...},
            ...
        ]

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of dicts with keys 'position' (tuple of x, y, z) and optional 'enabled'.
        If 'enabled' is present and False, the entry is filtered out.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the JSON format is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Hoop coordinate file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array at root level, got {type(data).__name__}")

    hoops = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {i} is not a dict: {entry}")

        # Skip disabled entries
        if not entry.get("enabled", True):
            continue

        pos = entry.get("position")
        if pos is None:
            raise ValueError(f"Entry {i} missing 'position' key")

        pos_list = list(pos) if hasattr(pos, "__iter__") else []
        if len(pos_list) != 3:
            raise ValueError(f"Entry {i} position is not length 3: {pos_list}")

        hoops.append({
            "position": tuple(float(v) for v in pos_list),
            "enabled": entry.get("enabled", True),
            "index": i,
        })

    return hoops


def move_to(
    arm: Any,
    x: float,
    y: float,
    z: float,
    yaw: float = 0.0,
    *,
    hover_height: float = 0.0,
    verbose: bool = False,
) -> bool:
    """
    Move the arm's gripper to a Cartesian position using hybrid analytic yaw + numeric IK.

    This function:
      1. Computes an analytic yaw if not provided (faces the target: atan2(y, x)).
      2. Uses PyBullet's numeric IK to solve for shoulder/elbow/wrist angles.
      3. Commands the arm via set_joint_positions.

    Args:
        arm: QArmBase instance from api.factory.make_qarm.
        x, y, z: Target position (meters) in robot base frame.
        yaw: Optional base yaw angle (radians). If 0 (default), computed analytically.
        hover_height: Vertical offset applied during motion. If > 0, arm moves with z
                     raised by this amount (useful for cartesian interpolation safety).
        verbose: If True, print debug information.

    Returns:
        True if the move succeeded; False if IK failed or arm unavailable.
    """
    env = getattr(arm, "env", None)
    if env is None:
        if verbose:
            print("[move_to] No env; cannot compute IK.")
        return False

    if p is None:
        if verbose:
            print("[move_to] PyBullet not available.")
        return False

    # Compute analytic yaw if not provided (point base at target)
    target_yaw = yaw if yaw != 0.0 else math.atan2(y, x)

    # Effective target with hover offset
    target_z = z + hover_height
    target_pos = (float(x), float(y), float(target_z))

    # Get IK target link (gripper base or end-effector)
    try:
        link_idx = getattr(env, "_gripper_base_link_index", None)
        if link_idx is None:
            idx_map = getattr(env, "link_name_to_index", {})
            link_idx = idx_map.get("GRIPPER_BASE") or idx_map.get("END-EFFECTOR")
        if link_idx is None:
            if verbose:
                print("[move_to] Could not find gripper/EE link for IK.")
            return False
    except Exception as e:
        if verbose:
            print(f"[move_to] Failed to locate IK target link: {e}")
        return False

    # Solve IK via PyBullet
    try:
        ik_solution = p.calculateInverseKinematics(
            env.robot_id,
            link_idx,
            target_pos,
            physicsClientId=env.client,
        )
        if not ik_solution:
            if verbose:
                print(f"[move_to] IK returned empty solution for {target_pos}.")
            return False
    except Exception as e:
        if verbose:
            print(f"[move_to] IK failed for target {target_pos}: {e}")
        return False

    # Map IK solution to arm joint ordering
    try:
        joint_indices = getattr(arm, "joint_order", None)
        if joint_indices is None:
            joint_indices = getattr(env, "movable_joint_indices", [])
        targets = [float(ik_solution[j]) for j in joint_indices]
    except Exception as e:
        if verbose:
            print(f"[move_to] Failed to map IK solution to joints: {e}")
        return False

    # Enforce analytic yaw (first joint)
    if targets:
        targets[0] = target_yaw

    # Command the arm
    try:
        arm.set_joint_positions(targets)
        if verbose:
            print(f"[move_to] Moved to ({x:.3f}, {y:.3f}, {z:.3f}) yaw={target_yaw:.3f} rad")
        return True
    except Exception as e:
        if verbose:
            print(f"[move_to] set_joint_positions failed: {e}")
        return False


def pick_hoop(
    arm: Any,
    x: float,
    y: float,
    z: float,
    *,
    approach_height: float = 0.10,
    gripper_closed_angle: float = 0.55,
    move_delay_s: float = 0.6,
    verbose: bool = False,
) -> bool:
    """
    Pick a hoop at the given Cartesian position.

    Sequence:
      1. Move to approach position (z + approach_height).
      2. Lower to grasp position (z).
      3. Close gripper.
      4. Lift back to approach position.

    Args:
        arm: QArmBase instance.
        x, y, z: Hoop center position (meters) in robot base frame.
        approach_height: Height above hoop to move to first (default 0.10 m = 10 cm).
        gripper_closed_angle: Gripper angle when fully closed (default 0.55 rad).
        move_delay_s: Time to wait between moves for simulation/gripper settling.
        verbose: If True, print debug info.

    Returns:
        True if all steps succeeded; False if any step failed.
    """
    if verbose:
        print(f"[pick_hoop] Starting pick at ({x:.3f}, {y:.3f}, {z:.3f})")

    # Move to approach
    if not move_to(arm, x, y, z, hover_height=approach_height, verbose=verbose):
        if verbose:
            print("[pick_hoop] Failed to reach approach position.")
        return False
    time.sleep(move_delay_s)

    # Lower to grasp (no hover)
    if not move_to(arm, x, y, z, hover_height=0.0, verbose=verbose):
        if verbose:
            print("[pick_hoop] Failed to lower to grasp position.")
        return False
    time.sleep(move_delay_s)

    # Close gripper
    try:
        arm.set_gripper_position(gripper_closed_angle)
        if verbose:
            print(f"[pick_hoop] Gripper closed (angle={gripper_closed_angle:.3f})")
    except Exception as e:
        if verbose:
            print(f"[pick_hoop] Failed to close gripper: {e}")
        return False
    time.sleep(move_delay_s)

    # Lift back to approach
    if not move_to(arm, x, y, z, hover_height=approach_height, verbose=verbose):
        if verbose:
            print("[pick_hoop] Failed to lift after grasp.")
        return False
    time.sleep(move_delay_s)

    if verbose:
        print(f"[pick_hoop] Successfully picked hoop at ({x:.3f}, {y:.3f}, {z:.3f})")
    return True


def place_hoop(
    arm: Any,
    px: float,
    py: float,
    pz: float,
    *,
    approach_height: float = 0.10,
    gripper_open_angle: float = 0.0,
    move_delay_s: float = 0.6,
    verbose: bool = False,
) -> bool:
    """
    Place a hoop (already grasped) onto a pole at the given position.

    Sequence:
      1. Move to approach position above pole (pz + approach_height).
      2. Lower to pole height (pz).
      3. Open gripper.
      4. Retreat upward to approach position.

    Args:
        arm: QArmBase instance.
        px, py, pz: Pole position (meters) in robot base frame.
        approach_height: Height above pole for approach/retreat (default 0.10 m = 10 cm).
        gripper_open_angle: Gripper angle when fully open (default 0.0 rad).
        move_delay_s: Time to wait between moves.
        verbose: If True, print debug info.

    Returns:
        True if all steps succeeded; False otherwise.
    """
    if verbose:
        print(f"[place_hoop] Starting place at pole ({px:.3f}, {py:.3f}, {pz:.3f})")

    # Move to approach
    if not move_to(arm, px, py, pz, hover_height=approach_height, verbose=verbose):
        if verbose:
            print("[place_hoop] Failed to reach approach position.")
        return False
    time.sleep(move_delay_s)

    # Lower to pole
    if not move_to(arm, px, py, pz, hover_height=0.0, verbose=verbose):
        if verbose:
            print("[place_hoop] Failed to lower to pole height.")
        return False
    time.sleep(move_delay_s)

    # Open gripper
    try:
        arm.set_gripper_position(gripper_open_angle)
        if verbose:
            print(f"[place_hoop] Gripper opened (angle={gripper_open_angle:.3f})")
    except Exception as e:
        if verbose:
            print(f"[place_hoop] Failed to open gripper: {e}")
        return False
    time.sleep(move_delay_s)

    # Retreat upward
    if not move_to(arm, px, py, pz, hover_height=approach_height, verbose=verbose):
        if verbose:
            print("[place_hoop] Failed to retreat after placing.")
        return False
    time.sleep(move_delay_s)

    if verbose:
        print(f"[place_hoop] Successfully placed hoop at pole ({px:.3f}, {py:.3f}, {pz:.3f})")
    return True


def sort_all_hoops(
    arm: Any,
    hoops: list[dict[str, Any]],
    pole: tuple[float, float, float],
    *,
    verbose: bool = True,
    stop_on_failure: bool = False,
) -> int:
    """
    Pick and place all hoops from the list onto a pole.

    This is the main orchestrator: for each hoop in the list, it calls
    pick_hoop and place_hoop in sequence. If a hoop pick/place fails,
    either skips it (default) or stops the entire sequence.

    Args:
        arm: QArmBase instance.
        hoops: List of hoop dicts with 'position' key (tuple of x, y, z).
        pole: Pole position (px, py, pz) as a tuple.
        verbose: If True, print detailed progress.
        stop_on_failure: If True, stop on first failure. If False, skip failed
                        hoops and continue with the rest.

    Returns:
        Number of hoops successfully placed.
    """
    if not hoops:
        if verbose:
            print("[sort_all_hoops] No hoops to sort.")
        return 0

    px, py, pz = float(pole[0]), float(pole[1]), float(pole[2])
    if verbose:
        print(f"[sort_all_hoops] Starting sort of {len(hoops)} hoop(s) to pole {pole}")

    placed_count = 0
    for i, hoop_info in enumerate(hoops):
        pos = hoop_info.get("position")
        if not pos or len(pos) != 3:
            if verbose:
                print(f"[sort_all_hoops] Hoop {i}: invalid position {pos}, skipping.")
            if stop_on_failure:
                break
            continue

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if verbose:
            print(f"\n[sort_all_hoops] === Hoop {i + 1}/{len(hoops)} ===")

        # Pick
        if not pick_hoop(arm, x, y, z, verbose=verbose):
            if verbose:
                print(f"[sort_all_hoops] Failed to pick hoop {i}.")
            if stop_on_failure:
                break
            continue

        # Place
        if not place_hoop(arm, px, py, pz, verbose=verbose):
            if verbose:
                print(f"[sort_all_hoops] Failed to place hoop {i}.")
            if stop_on_failure:
                break
            continue

        placed_count += 1
        if verbose:
            print(f"[sort_all_hoops] Hoop {i + 1} placed successfully ({placed_count}/{len(hoops)})")

    if verbose:
        print(f"\n[sort_all_hoops] Completed: {placed_count}/{len(hoops)} hoops placed.")
    return placed_count


def main() -> None:
    """Demo: load hoops from JSON and sort them onto a pole."""
    from api.factory import make_qarm

    print("[hoop_sorter] Initializing QArm in simulation mode...")
    arm = make_qarm(mode="sim", gui=False, auto_step=True)
    arm.home()
    print("[hoop_sorter] Arm ready.")

    # Load hoops from the demo JSON file
    try:
        demo_dir = Path(__file__).resolve().parent
        json_file = demo_dir / "hoop_positions.json"
        hoops = load_hoop_coordinates(json_file)
        print(f"[hoop_sorter] Loaded {len(hoops)} hoop(s) from {json_file}")
    except Exception as e:
        print(f"[hoop_sorter] Failed to load hoops: {e}")
        return

    # Define pole position (example: center of the board, at hoop height)
    pole_pos = (0.0, -0.35, 0.12)
    print(f"[hoop_sorter] Pole target: {pole_pos}")

    # Sort all hoops (pick and place onto pole)
    try:
        placed = sort_all_hoops(arm, hoops, pole_pos, verbose=True, stop_on_failure=False)
        print(f"[hoop_sorter] Sorting complete: {placed}/{len(hoops)} hoops placed.")
    except Exception as e:
        print(f"[hoop_sorter] Error during sorting: {e}")
    finally:
        print("[hoop_sorter] Returning to home...")
        arm.home()
        print("[hoop_sorter] Done.")


if __name__ == "__main__":
    main()
