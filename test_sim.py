"""
Friendly starter template for the QArm simulator.

Use this file as your sandbox: edit one section at a time, read the comments,
and try things out as you learn.

Run it:
    python -m demos.blank_sim
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from pathlib import Path
from typing import Iterable
import json
import math

try:
    import pybullet as p
except Exception:
    p = None

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase
from demos._shared import run_with_viewer

# ---------------------------------------------------------------------------
# Quick knobs to tweak the experience
# ---------------------------------------------------------------------------
MODE = "sim"  # keep "sim" until you have hardware; use "mirror" for hardware+sim
USE_PANDA_VIEWER = True  # Panda3D window that shows the arm; set False for console only
USE_PYBULLET_GUI = False  # Bullet's debug sliders (rarely needed)
TIME_STEP = 1.0 / 240.0
HEADLESS_SECONDS = 10.0  # how long to run when no viewer is open
SHOW_VIEWER_SLIDERS = False
RELOAD_MESHES = False


def _launch_viewer(arm: QArmBase) -> None:
    """Open the Panda3D viewer. Beginners: just leave this alone."""
    env = getattr(arm, "env", None)
    if env is None:
        print("[Blank] No simulator attached; viewer cannot start.")
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


def _headless_spin(env: object) -> None:
    """Keep physics running without graphics (useful for scripts/tests)."""
    if not hasattr(env, "step") or not hasattr(env, "time_step"):
        print("[Blank] No environment to step. Enable the simulator first.")
        return
    steps = int(HEADLESS_SECONDS / env.time_step)
    print(f"[Blank] Stepping headless for {HEADLESS_SECONDS:.1f}s ({steps} steps)...")
    for _ in range(steps):
        env.step()
        time.sleep(env.time_step)
    print("[Blank] Done stepping; press Ctrl+C to exit or start scripting moves.")


def main() -> None:
    """Entry point. Follow the comments below to add your own code."""
    mode = MODE.lower()
    mirror_mode = mode == "mirror"
    effective_mode = "hardware" if mirror_mode else mode
    auto_step = not USE_PANDA_VIEWER  # viewer updates the sim when it's open
    print(f"[Blank] Connecting to QArm in {mode} mode with joint order {DEFAULT_JOINT_ORDER}")
    arm = make_qarm(
        mode=effective_mode,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        time_step=TIME_STEP,
        auto_step=auto_step,
        mirror_sim=mirror_mode,
    )

    env = getattr(arm, "env", None)
    if env is not None:
        env.reset()  # start from a zeroed pose

    # Attempt to spawn demo hoops before the viewer starts so the viewer
    # picks them up during its initial setup (match demos/hoop_segments.py behavior).
    try:
        import demos.hoop_segments as _hoop_demo

        try:
            # snap first green hoop to accent if available
            _hoop_demo._place_first_green_over_accent(arm)
        except Exception:
            pass

        try:
            _hoop_demo.add_hoops(arm, _hoop_demo.DEFAULT_HOOP_DEFS)
            print("[TestSim] Spawned demo hoops via demos.hoop_segments before viewer init.")
        except Exception as e:
            print(f"[TestSim] Failed to spawn demo hoops early: {e}")
    except Exception:
        # If import fails, fall back to student_script spawning later.
        pass

    try:
        if USE_PANDA_VIEWER:
            print("[Blank] Viewer opening. Running your script while the window is visible.")
            # Use a worker thread so `student_script` runs at the same time as the viewer.
            # Threads let one part of your program keep working (running commands) while
            # another part (the Panda window) stays responsive on the main thread.
            import threading

            worker = threading.Thread(target=student_script, args=(arm,), daemon=True)
            worker.start()
            _launch_viewer(arm)
            worker.join()
        elif env is not None:
            student_script(arm)
            _headless_spin(env)
        else:
            print("[Blank] No simulator to run; check MODE.")
    except KeyboardInterrupt:
        print("\n[Blank] Stopping minimal sim.")
    finally:
        try:
            arm.home()
        except Exception:
            pass
        disconnect = getattr(arm, "disconnect", None)
        if callable(disconnect):
            try:
                disconnect()
            except Exception:
                pass


def student_script(arm: QArmBase) -> None:
    """
    BEGINNER PLAYGROUND: put your experiments here.

    This function runs after the viewer is on-screen (when enabled), so you can see everything.
    Uncomment one idea at a time or replace them with your own.
    """

    env = getattr(arm, "env", None)
    # If hoops were spawned earlier (e.g., by main using the demo helper), skip re-spawning.
    try:
        if env is not None and getattr(env, "kinematic_objects", None):
            print("[TestSim] Kinematic objects already present; skipping hoop spawn in student_script.")
            return
    except Exception:
        pass

    time.sleep(4.0)  # wait a moment for things to settle

    # ---------- Hoop spawning helpers (merged from demos/hoop_segments.py) ----------
    DEMO_DIR = Path(__file__).resolve().parent / "demos"
    MODEL_DIR = DEMO_DIR / "models"

    HOOP_MESH = MODEL_DIR / "hoop.stl"
    HOOP_SEGMENT = MODEL_DIR / "hoop-segment.stl"
    HOOP_COLLISION_SEGMENTS = {
        "mesh_path": HOOP_SEGMENT,
        "radius": 68.0 / 2.0,
        "yaw_step_deg": 29.9,
        "count": 12,
    }

    SPAWN_MULTIPLE_HOOPS = True
    BOARD_SURFACE_Z = 0.08

    def _make_board_hoop_defs(
        *,
        greens: int = 8,
        purples: int = 8,
        x_span: float = 0.56,
        y_front: float = -0.25,
        y_back: float = -0.35,
        z: float = BOARD_SURFACE_Z,
        scale: float = 0.001,
    ):
        defs: list[dict] = []
        def xs(count: int) -> list[float]:
            if count == 1:
                return [0.0]
            step = x_span / (count - 1)
            start = -x_span / 2.0
            return [start + i * step for i in range(count)]

        green_xs = xs(greens)
        purple_xs = xs(purples)

        green_color = (0.1, 0.9, 0.1, 1.0)
        purple_color = (0.6, 0.2, 0.8, 1.0)

        radius_mm = HOOP_COLLISION_SEGMENTS.get("radius", 0.0)
        radius_m = (radius_mm * 0.001) * scale
        center_z = z + radius_m

        for x in green_xs:
            defs.append({
                "position": (x, y_front, center_z),
                "orientation_euler_deg": (0.0, 0.0, 0.0),
                "scale": scale,
                "rgba": green_color,
                "enabled": True,
            })

        for x in purple_xs:
            defs.append({
                "position": (x, y_back, center_z),
                "orientation_euler_deg": (0.0, 0.0, 0.0),
                "scale": scale,
                "rgba": purple_color,
                "enabled": True,
            })

        return defs

    DEFAULT_HOOP_DEFS = _make_board_hoop_defs()

    HOOP_POS_FILE = DEMO_DIR / "hoop_positions.json"
    if HOOP_POS_FILE.exists():
        try:
            with HOOP_POS_FILE.open("r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                def _norm(d: dict) -> dict:
                    out = dict(d)
                    if "position" in out and isinstance(out["position"], list):
                        out["position"] = tuple(out["position"])
                    if "orientation_euler_deg" in out and isinstance(out["orientation_euler_deg"], list):
                        out["orientation_euler_deg"] = tuple(out["orientation_euler_deg"])
                    if "rgba" in out and isinstance(out["rgba"], list):
                        out["rgba"] = tuple(out["rgba"])
                    if "enabled" not in out:
                        out["enabled"] = True
                    return out

                DEFAULT_HOOP_DEFS = [_norm(x) for x in loaded]
                enabled_count = sum(1 for d in DEFAULT_HOOP_DEFS if d.get("enabled", True))
                print(f"[TestSim] Loaded {len(DEFAULT_HOOP_DEFS)} hoop definitions from {HOOP_POS_FILE} ({enabled_count} enabled)")
            else:
                print(f"[TestSim] {HOOP_POS_FILE} does not contain a list; using defaults.")
        except Exception as e:
            print(f"[TestSim] Failed to load {HOOP_POS_FILE}: {e}; using defaults.")

    def add_hoop(
        arm: QArmBase,
        *,
        mesh_path: Path | None = None,
        collision_segments=None,
        position=(0.0, -0.30, 0.08),
        orientation_euler_deg=(0.0, 0.0, 20.0),
        scale=0.001,
        rgba=(0.1, 0.9, 0.1, 1.0),
        mass: float = 0.1,
    ):
        env = getattr(arm, "env", None)
        if env is None or not hasattr(env, "add_kinematic_object"):
            print("[Hoop] Simulator backend not available; skipping hoop spawn.")
            return

        mesh_path = mesh_path or HOOP_MESH
        collision_segments = collision_segments or HOOP_COLLISION_SEGMENTS
        if not mesh_path.exists() or not HOOP_SEGMENT.exists():
            print(f"[Hoop] Missing hoop meshes under {MODEL_DIR}.")
            return

        env.add_kinematic_object(
            mesh_path=mesh_path,
            position=position,
            orientation_euler_deg=orientation_euler_deg,
            scale=scale,
            collision_segments=collision_segments,
            mass=mass,
            force_convex_for_dynamic=True,
            rgba=rgba,
            lateral_friction=1.0,
            rolling_friction=0.05,
            spinning_friction=0.05,
            restitution=0.0,
        )
        print(f"[Hoop] Spawned hoop at {position} with orientation {orientation_euler_deg}.")

    def add_hoops(arm: QArmBase, definitions: Iterable[dict]) -> None:
        enabled_defs = [d for d in definitions if d.get("enabled", True)]
        skipped = len(definitions) - len(enabled_defs)
        if skipped:
            print(f"[Hoop] Skipping {skipped} disabled hoop definition(s).")

        for d in enabled_defs:
            try:
                kwargs = dict(d)
                kwargs.pop("enabled", None)
                add_hoop(arm, **kwargs)
            except TypeError as e:
                print(f"[Hoop] Invalid hoop definition {d}: {e}")

    def _find_accent_world_positions(env, accent_stem: str = "greenaccent") -> list[tuple[float, float, float]]:
        if p is None:
            return []
        floor_id = getattr(env, "floor_id", None)
        if floor_id is None:
            return []
        try:
            shapes = p.getVisualShapeData(floor_id, physicsClientId=env.client)
        except Exception:
            return []
        try:
            base_pos, base_orn = p.getBasePositionAndOrientation(floor_id, physicsClientId=env.client)
        except Exception:
            base_pos, base_orn = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        out: list[tuple[float, float, float]] = []
        for shape in shapes:
            filename = shape[4]
            if not filename:
                continue
            stem = Path(filename).stem.lower()
            if accent_stem.lower() not in stem:
                continue
            local_pos = shape[5]
            local_orn = shape[6]
            try:
                world_pos, _ = p.multiplyTransforms(base_pos, base_orn, local_pos, local_orn)
                out.append((float(world_pos[0]), float(world_pos[1]), float(world_pos[2])))
            except Exception:
                continue
        return out

    def _place_first_green_over_accent(arm: QArmBase) -> bool:
        env = getattr(arm, "env", None)
        if env is None:
            return False
        base_assets = getattr(env, 'base_assets', None)
        accent_name = Path(base_assets.green_accent_mesh).stem if base_assets is not None else 'greenaccent'
        accents = _find_accent_world_positions(env, accent_stem=accent_name)
        if not accents:
            return False
        accent_pos = accents[0]
        for d in DEFAULT_HOOP_DEFS:
            if not d.get("enabled", True):
                continue
            rgba = d.get("rgba")
            if not rgba:
                continue
            try:
                if float(rgba[1]) > 0.6:
                    radius_mm = HOOP_COLLISION_SEGMENTS.get("radius", 0.0)
                    scale = d.get("scale", 0.001)
                    center_z = BOARD_SURFACE_Z + (radius_mm * 0.001) * float(scale)
                    d["position"] = (accent_pos[0], accent_pos[1], center_z)
                    print(f"[Hoop] Moved first enabled green hoop to accent at {d['position']}")
                    return True
            except Exception:
                continue
        return False

    # ---------- End hoop helpers ----------

    # Try to snap the first green hoop to the green accent/pole, then spawn enabled hoops
    try:
        _place_first_green_over_accent(arm)
    except Exception:
        pass

    if SPAWN_MULTIPLE_HOOPS:
        add_hoops(arm, DEFAULT_HOOP_DEFS)

    # --- Example 1: print the current joint angles (yaw, shoulder, elbow, wrist)
    # print("Current joints:", arm.get_joint_positions())

    # --- Example 2: nudge the elbow slightly, then print the new values
    # joints = arm.get_joint_positions()
    # joints[2] += 0.5  # index 2 is the elbow
    # arm.set_joint_positions(joints)
    # print("Moved joints:", arm.get_joint_positions())

    # --- Example 3: alternate between two poses a few times
    # pose_a = [0.0, 0.3, -0.3, 0.0]
    # pose_b = [0.0, 0.6, -0.6, 0.0]
    # for i in range(5):
    #     arm.set_joint_positions(pose_a)
    #     time.sleep(0.5)
    #     arm.set_joint_positions(pose_b)
    #     time.sleep(0.5)

    # Remove the `pass` when you start adding your own code.
    pass


if __name__ == "__main__":
    main()
