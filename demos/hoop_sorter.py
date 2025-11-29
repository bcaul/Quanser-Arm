"""
Hoop Sorting Module for Quanser QArm (4-DOF).

    Entry point: load hoops, run sorting with viewer, and enable color-based pole routing.
and autonomously pick and place each hoop onto a pole using inverse kinematics
and gripper control.

Key functions:
  - load_hoop_coordinates(filepath): Load hoop positions from JSON.
  - move_to(arm, x, y, z, yaw=0): Cartesian motion using hybrid analytic yaw + PyBullet IK.
    # Pole coordinates (user-provided)
    green_pole = (-0.09, -0.57, 0.195)
    purple_pole = (0.3, -0.09, 0.195)

  - pick_hoop(arm, x, y, z): Move above, lower, grasp, and lift a hoop.
    run_sorting_with_viewer(hoops, green_pole=green_pole, purple_pole=purple_pole)
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
    """Slowly move vertically between two Z heights while keeping X/Y fixed.

    This interpolates intermediate Z targets and calls `move_to` for each
    so the arm descends/ascends approximately vertically instead of a
    single IK solution that may produce diagonal plunges.
    """
    if steps < 1:
        steps = 1
    target_yaw = yaw if yaw is not None else math.atan2(y, x)
    for i in range(1, steps + 1):
        alpha = i / float(steps)
        z_i = float(z_from) + (float(z_to) - float(z_from)) * alpha
        ok = move_to(arm, x, y, z_i, yaw=target_yaw, hover_height=0.0, verbose=verbose)
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
    approach_height: float = 0.08,
    grasp_z_offset: float = 0.002,
    gripper_closed_angle: float = 0.55,
    move_delay_s: float = 1.0,
    gripper_settle_s: float = 0.8,
    max_attempts: int = 4,
    nudge_xy: float = 0.004,
    rest_home_after_pick: bool = False,
    verbose: bool = False,
) -> bool:
    """
    Pick a hoop at the given Cartesian position.

    Sequence:
      1. Move to approach position (z + approach_height).
      2. Lower to grasp position (z + grasp_z_offset, just above the hoop center).
      3. Close gripper with settling time.
      4. Lift back to approach position.

    Args:
        arm: QArmBase instance.
        x, y, z: Hoop center position (meters) in robot base frame.
        approach_height: Height above hoop to move to first (default 0.08 m = 8 cm).
        grasp_z_offset: Small offset above the hoop center for grasp (default 0.002 m).
                       Hoops rest at z=0.108; moving to z+0.002 positions gripper
                       just above the hoop's contact point for optimal grasping.
        gripper_closed_angle: Gripper angle when fully closed (default 0.55 rad).
        move_delay_s: Time to wait after each move for arm settling.
        gripper_settle_s: Additional time to allow gripper to fully close and lock hoop.
        verbose: If True, print debug info.

    Returns:
        True if all steps succeeded; False if any step failed.
    """
    if verbose:
        print(f"[pick_hoop] Starting pick at ({x:.3f}, {y:.3f}, {z:.3f})")

    # We'll attempt multiple small retries with slight XY nudges and small lower offsets
    env = getattr(arm, "env", None)
    # Precompute nudge offsets (center + 4-direction nudges)
    nudges = [(0.0, 0.0), (nudge_xy, 0.0), (-nudge_xy, 0.0), (0.0, nudge_xy), (0.0, -nudge_xy)]
    z_offsets = [0.0, -0.002, -0.004, -0.006]

    # Ensure gripper starts open
    try:
        arm.set_gripper_position(0.0)
    except Exception:
        pass

    for attempt in range(1, max_attempts + 1):
        if verbose:
            print(f"[pick_hoop] Attempt {attempt}/{max_attempts} for hoop at ({x:.3f},{y:.3f},{z:.3f})")

        for dx, dy in nudges:
            tx, ty = x + dx, y + dy

            # Move to approach above the nudged target
            target_yaw = math.atan2(ty, tx)
            if not move_to(arm, tx, ty, z, yaw=target_yaw, hover_height=approach_height, verbose=verbose):
                if verbose:
                    print(f"[pick_hoop] Failed to reach approach at nudge ({dx:.3f},{dy:.3f}).")
                continue
            time.sleep(move_delay_s)

            # Try a series of slightly lower grasp offsets
            for z_off in z_offsets:
                grasp_z = z + grasp_z_offset + z_off
                # Do a slow, vertical descent to the grasp height to avoid diagonal plunges
                if not slow_vertical_move(arm, tx, ty, z + approach_height, grasp_z, steps=6, delay=move_delay_s / 2.0, yaw=target_yaw, verbose=verbose):
                    if verbose:
                        print(f"[pick_hoop] Failed to lower to grasp z={grasp_z:.3f}.")
                    continue
                time.sleep(move_delay_s)

                # Close gripper
                try:
                    arm.set_gripper_position(gripper_closed_angle)
                    if verbose:
                        print(f"[pick_hoop] Gripper closing (angle={gripper_closed_angle:.3f})")
                except Exception as e:
                    if verbose:
                        print(f"[pick_hoop] Failed to close gripper: {e}")
                    continue
                time.sleep(gripper_settle_s)

                # Lift back to approach to let env detect lock
                # Lift back to approach height slowly
                if not slow_vertical_move(arm, tx, ty, grasp_z, z + approach_height, steps=6, delay=move_delay_s / 2.0, yaw=target_yaw, verbose=verbose):
                    if verbose:
                        print("[pick_hoop] Failed to lift after grasp attempt.")
                    continue
                time.sleep(move_delay_s)

                # Check if environment has locked a grasped body
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
                    # Optionally return the arm to home to give room for transit
                    if rest_home_after_pick:
                        try:
                            if verbose:
                                print("[pick_hoop] Returning arm to home after successful grasp.")
                            arm.home()
                        except Exception:
                            pass
                        time.sleep(0.6)
                    return True

                # If not grasped, open gripper and try next offset/nudge
                try:
                    arm.set_gripper_position(0.0)
                    if verbose:
                        print("[pick_hoop] Gripper opened to retry.")
                except Exception:
                    pass
                time.sleep(0.12)

        # End of nudges; small pause between attempts
        time.sleep(0.15)

    # All attempts exhausted
    if verbose:
        print(f"[pick_hoop] All attempts failed to pick hoop at ({x:.3f}, {y:.3f}, {z:.3f})")
    return False


def place_hoop(
    arm: Any,
    px: float,
    py: float,
    pz: float,
    *,
    approach_height: float = 0.12,
    release_offset: float = 0.01,
    gripper_open_angle: float = 0.0,
    move_delay_s: float = 0.8,
    transit_z: float | None = None,
    slow_descent: bool = True,
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

    # Move to approach above pole
    if not move_to(arm, px, py, pz, hover_height=approach_height, verbose=verbose):
        if verbose:
            print("[place_hoop] Failed to reach approach position.")
        return False
    time.sleep(move_delay_s)

    # Lower to a small offset above the pole and release there (hover then drop)
    release_z = pz + release_offset
    # If a transit_z is provided, assume the arm is already at transit_z above XY and do a slow vertical descent
    if transit_z is not None and slow_descent:
        if not slow_vertical_move(arm, px, py, transit_z, release_z, steps=8, delay=move_delay_s / 2.0, yaw=math.atan2(py, px), verbose=verbose):
            if verbose:
                print("[place_hoop] Failed slow descent to release height.")
            return False
    else:
        if not move_to(arm, px, py, release_z, hover_height=0.0, verbose=verbose):
            if verbose:
                print("[place_hoop] Failed to lower to release height.")
            return False
    time.sleep(move_delay_s)

    # Open gripper to drop hoop while hovering above pole
    try:
        arm.set_gripper_position(gripper_open_angle)
        if verbose:
            print(f"[place_hoop] Gripper opened (angle={gripper_open_angle:.3f}) at z={release_z:.3f}")
    except Exception as e:
        if verbose:
            print(f"[place_hoop] Failed to open gripper: {e}")
        return False
    time.sleep(move_delay_s)

    # Retreat upward back to transit_z or approach height
    retreat_z = transit_z if transit_z is not None else (pz + approach_height)
    if retreat_z is not None:
        if not slow_vertical_move(arm, px, py, release_z, retreat_z, steps=6, delay=move_delay_s / 2.0, yaw=math.atan2(py, px), verbose=verbose):
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
    pole: tuple[float, float, float] | None = None,
    *,
    green_pole: tuple[float, float, float] | None = None,
    purple_pole: tuple[float, float, float] | None = None,
    verbose: bool = True,
    stop_on_failure: bool = False,
) -> int:
    """
    Pick and place all hoops from the list onto appropriate poles.

    If both green_pole and purple_pole are provided, routes hoops by color:
      - Green hoops (rgba[1] > 0.6) go to green_pole
      - Purple hoops go to purple_pole
    If only pole is provided, all hoops go to the same pole.

    This is the main orchestrator: for each hoop in the list, it calls
    pick_hoop and place_hoop in sequence. If a hoop pick/place fails,
    either skips it (default) or stops the entire sequence.

    Args:
        arm: QArmBase instance.
        hoops: List of hoop dicts with 'position' key (tuple of x, y, z) and optional 'rgba'.
        pole: Default pole position (px, py, pz) as a tuple. Used if green/purple poles not specified.
        green_pole: Green pole position tuple (px, py, pz). If provided with purple_pole, enables color routing.
        purple_pole: Purple pole position tuple (px, py, pz). If provided with green_pole, enables color routing.
        verbose: If True, print detailed progress.
        stop_on_failure: If True, stop on first failure. If False, skip failed hoops and continue.

    Returns:
        Number of hoops successfully placed.
    """
    if not hoops:
        if verbose:
            print("[sort_all_hoops] No hoops to sort.")
        return 0

    # Determine which pole routing to use
    use_color_routing = green_pole is not None and purple_pole is not None
    default_pole = pole or (0.0, -0.35, 0.12)

    if verbose:
        if use_color_routing:
            print(f"[sort_all_hoops] Color-based routing enabled:")
            print(f"  Green pole: {green_pole}")
            print(f"  Purple pole: {purple_pole}")
        else:
            print(f"[sort_all_hoops] Single pole mode: {default_pole}")
        print(f"[sort_all_hoops] Starting sort of {len(hoops)} hoop(s)")

    placed_count = 0
    # Ensure we start from a known home position
    try:
        if hasattr(arm, "home"):
            if verbose:
                print("[sort_all_hoops] Moving arm to home before starting.")
            arm.home()
            time.sleep(0.6)
    except Exception:
        pass
    for i, hoop_info in enumerate(hoops):
        pos = hoop_info.get("position")
        if not pos or len(pos) != 3:
            if verbose:
                print(f"[sort_all_hoops] Hoop {i}: invalid position {pos}, skipping.")
            if stop_on_failure:
                break
            continue

        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        # Determine target pole based on color if routing enabled
        target_pole = default_pole
        hoop_color = None
        if use_color_routing:
            rgba = hoop_info.get("rgba")
            if rgba and len(rgba) >= 3:
                # Heuristic: if green channel (index 1) > 0.6, treat as green hoop
                g_channel = float(rgba[1])
                if g_channel > 0.6:
                    target_pole = green_pole
                    hoop_color = "green"
                else:
                    target_pole = purple_pole
                    hoop_color = "purple"

        px, py, pz = float(target_pole[0]), float(target_pole[1]), float(target_pole[2])

        if verbose:
            color_str = f" ({hoop_color})" if hoop_color else ""
            print(f"\n[sort_all_hoops] === Hoop {i + 1}/{len(hoops)}{color_str} ===")

        # Pick (with slower motion and rest-home)
        if not pick_hoop(
            arm,
            x,
            y,
            z,
            verbose=verbose,
            move_delay_s=1.0,
            gripper_settle_s=0.8,
            max_attempts=4,
            nudge_xy=0.004,
            rest_home_after_pick=False,
        ):
            if verbose:
                print(f"[sort_all_hoops] Failed to pick hoop {i}.")
            if stop_on_failure:
                break
            continue

        # Place (raise to transit height, move horizontally, then slow descend and release)
        transit_z = max(0.20, pz + 0.08)
        # Raise vertically from hoop approach to transit_z
        try:
            if verbose:
                print(f"[sort_all_hoops] Raising to transit height {transit_z:.3f} before moving to pole.")
            slow_vertical_move(arm, x, y, z + 0.08, transit_z, steps=8, delay=0.12, yaw=math.atan2(y, x), verbose=verbose)
        except Exception:
            pass

        # Move horizontally to pole at transit height
        if not move_to(arm, px, py, transit_z, yaw=math.atan2(py, px), hover_height=0.0, verbose=verbose):
            if verbose:
                print(f"[sort_all_hoops] Failed to move to pole XY at transit height.")
            if stop_on_failure:
                break
            continue
        time.sleep(0.4)

        # Now perform slow vertical placement from transit_z down to release_z
        if not place_hoop(
            arm,
            px,
            py,
            pz,
            verbose=verbose,
            approach_height=0.12,
            release_offset=0.01,
            move_delay_s=0.8,
            transit_z=transit_z,
            slow_descent=True,
        ):
            if verbose:
                print(f"[sort_all_hoops] Failed to place hoop {i}.")
            if stop_on_failure:
                break
            continue

        placed_count += 1
        if verbose:
            print(f"[sort_all_hoops] Hoop {i + 1} placed successfully ({placed_count}/{len(hoops)})")
        # Return to home between cycles so the arm starts from a known pose
        try:
            if verbose:
                print("[sort_all_hoops] Returning arm to home after placing.")
            arm.home()
        except Exception:
            pass
        time.sleep(0.6)

    if verbose:
        print(f"\n[sort_all_hoops] Completed: {placed_count}/{len(hoops)} hoops placed.")
    return placed_count


def run_sorting_with_viewer(
    hoops: list[dict[str, Any]],
    pole: tuple[float, float, float],
    green_pole: tuple[float, float, float] | None = None,
    purple_pole: tuple[float, float, float] | None = None,
    verbose: bool = True,
) -> None:
    """Orchestrator that runs the arm in simulation with Panda3D viewer for color-based hoop sorting.

    This function sets up the arm, spawns hoops, and runs sorting while the
    viewer is open so you can watch the arm move in real-time.

    Args:
        hoops: List of hoop coordinate dicts.
        pole: Pole position tuple (px, py, pz).
        green_pole: Green pole position for color-based routing.
        purple_pole: Purple pole position for color-based routing.
        verbose: If True, print detailed progress.
    """
    from api.factory import make_qarm
    from types import SimpleNamespace
    from sim.panda_viewer import PandaArmViewer, PhysicsBridge
    import threading

    print("[run_sorting_with_viewer] Initializing QArm in simulation mode...")
    # Create arm WITHOUT auto_step so the viewer controls stepping
    arm = make_qarm(mode="sim", gui=False, real_time=False, auto_step=False)
    arm.home()
    print("[run_sorting_with_viewer] Arm ready.")

    # Spawn hoops before viewer starts (so viewer picks them up at setup)
    try:
        import demos.hoop_segments as _hoop_demo
        try:
            _hoop_demo._place_first_green_over_accent(arm)
        except Exception:
            pass
        try:
            _hoop_demo.add_hoops(arm, _hoop_demo.DEFAULT_HOOP_DEFS)
            print("[run_sorting_with_viewer] Demo hoops spawned.")
        except Exception as e:
            print(f"[run_sorting_with_viewer] Warning: could not spawn demo hoops: {e}")
    except Exception:
        pass

    def sorting_worker() -> None:
        """Worker thread that runs the sorting sequence."""
        try:
            print("[sorting_worker] Starting hoop sorting...")
            placed = sort_all_hoops(
                arm,
                hoops,
                pole,
                green_pole=green_pole,
                purple_pole=purple_pole,
                verbose=verbose,
                stop_on_failure=False,
            )
            print(f"[sorting_worker] Sorting complete: {placed}/{len(hoops)} hoops placed.")
        except Exception as e:
            print(f"[sorting_worker] Error during sorting: {e}")
        finally:
            try:
                print("[sorting_worker] Returning to home...")
                arm.home()
            except Exception:
                pass

    def launch_viewer() -> None:
        """Launch the Panda3D viewer (blocks until window closes)."""
        try:
            env = getattr(arm, "env", None)
            if env is None:
                print("[launch_viewer] No env; viewer unavailable.")
                return
            viewer_args = SimpleNamespace(
                time_step=env.time_step,
                hide_base=False,
                hide_accents=False,
                probe_base_collision=False,
                show_sliders=False,
                reload_meshes=False,
            )
            bridge = PhysicsBridge(time_step=env.time_step, env=env, reset=False)
            PandaArmViewer(bridge, viewer_args).run()
        except Exception as e:
            print(f"[launch_viewer] Viewer error: {e}")

    # Start sorting in background thread
    worker = threading.Thread(target=sorting_worker, daemon=True)
    worker.start()

    # Launch viewer (blocks until closed)
    launch_viewer()

    # Wait for sorting to finish
    worker.join(timeout=120.0)
    print("[run_sorting_with_viewer] Done.")


def main() -> None:
    """Demo: load hoops from JSON and sort them onto color-based poles with viewer."""
    print("[hoop_sorter] Initializing...")

    # Load hoops from the demo JSON file
    try:
        demo_dir = Path(__file__).resolve().parent
        json_file = demo_dir / "hoop_positions.json"
        hoops = load_hoop_coordinates(json_file)
        print(f"[hoop_sorter] Loaded {len(hoops)} hoop(s) from {json_file}")
    except Exception as e:
        print(f"[hoop_sorter] Failed to load hoops: {e}")
        return

    # Define pole positions (user-provided)
    default_pole = (0.0, -0.35, 0.12)
    green_pole = (-0.09, -0.57, 0.195)
    purple_pole = (0.3, -0.09, 0.195)
    print(f"[hoop_sorter] Pole targets:")
    print(f"  Green pole:  {green_pole}")
    print(f"  Purple pole: {purple_pole}")

    # Run with viewer
    try:
        run_sorting_with_viewer(
            hoops,
            default_pole,
            green_pole=green_pole,
            purple_pole=purple_pole,
            verbose=True,
        )
    except Exception as e:
        print(f"[hoop_sorter] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
