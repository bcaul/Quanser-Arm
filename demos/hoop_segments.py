"""
Hoop demo using collision segments for a more realistic ring.

- Spawns one hoop mesh plus a compound collision ring built from hoop segments.
- Uses the same ring parameters as the other hoop demos.
- Panda3D viewer opens by default so you can see the hoop placement.

Run it:
    python -m demos.hoop_segments
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable
import json

from api.factory import make_qarm
from common.qarm_base import QArmBase

MODE = "sim"
USE_PANDA_VIEWER = True
USE_PYBULLET_GUI = False
TIME_STEP = 1.0 / 240.0
MODEL_DIR = Path(__file__).resolve().parent / "models"

# Mesh & collision constants (must be defined before helpers that use them)
HOOP_MESH = MODEL_DIR / "hoop.stl"
HOOP_SEGMENT = MODEL_DIR / "hoop-segment.stl"
HOOP_COLLISION_SEGMENTS = {
    "mesh_path": HOOP_SEGMENT,
    "radius": 68.0 / 2.0,  # mm ring diameter -> 34 mm radius before scaling
    "yaw_step_deg": 29.9,
    "count": 12,
}
# Toggle to spawn multiple hoops and a default list of hoop definitions.
SPAWN_MULTIPLE_HOOPS = True

# Board surface height used by other demos (meters)
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
    """Return a list of hoop definition dicts positioned on the board surface.

    Layout: two rows (front/back) each with `greens` or `purples` respectively,
    spread evenly across `x_span` (meters).
    """
    defs: list[dict] = []
    # Evenly spaced x positions centered at 0
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

    # Compute center z so the bottom of the hoop sits on the board surface.
    # HOOP_COLLISION_SEGMENTS['radius'] is given in mm; convert to meters
    # and multiply by the mesh scale (which typically converts mm->m).
    radius_mm = HOOP_COLLISION_SEGMENTS.get("radius", 0.0)
    radius_m = (radius_mm * 0.001) * scale
    center_z = z + radius_m

    for x in green_xs:
        defs.append({
            "position": (x, y_front, center_z),
            "orientation_euler_deg": (0.0, 0.0, 0.0),
            "scale": scale,
            "rgba": green_color,
        })

    for x in purple_xs:
        defs.append({
            "position": (x, y_back, center_z),
            "orientation_euler_deg": (0.0, 0.0, 0.0),
            "scale": scale,
            "rgba": purple_color,
        })

    return defs


DEFAULT_HOOP_DEFS = _make_board_hoop_defs()

# Allow optional override via a JSON file placed next to this demo file. The
# file should contain a JSON array of hoop definition objects (same kwargs as
# passed to `add_hoop`). Example entry:
#   {"position": [0.1, -0.3, 0.12], "rgba": [1,0,0,1], "scale": 0.001}
HOOP_POS_FILE = Path(__file__).resolve().parent / "hoop_positions.json"
if HOOP_POS_FILE.exists():
    try:
        with HOOP_POS_FILE.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, list):
            # Convert lists to tuples for positions and colors if necessary
            def _norm(d: dict) -> dict:
                out = dict(d)
                if "position" in out and isinstance(out["position"], list):
                    out["position"] = tuple(out["position"])
                if "orientation_euler_deg" in out and isinstance(out["orientation_euler_deg"], list):
                    out["orientation_euler_deg"] = tuple(out["orientation_euler_deg"])
                if "rgba" in out and isinstance(out["rgba"], list):
                    out["rgba"] = tuple(out["rgba"])
                # Ensure an 'enabled' flag exists so entries can be toggled in JSON
                if "enabled" not in out:
                    out["enabled"] = True
                return out

            DEFAULT_HOOP_DEFS = [_norm(x) for x in loaded]
            enabled_count = sum(1 for d in DEFAULT_HOOP_DEFS if d.get("enabled", True))
            print(f"[Hoop] Loaded {len(DEFAULT_HOOP_DEFS)} hoop definitions from {HOOP_POS_FILE} ({enabled_count} enabled)")
        else:
            print(f"[Hoop] {HOOP_POS_FILE} does not contain a list; using defaults.")
    except Exception as e:
        print(f"[Hoop] Failed to load {HOOP_POS_FILE}: {e}; using defaults.")

HOOP_MESH = MODEL_DIR / "hoop.stl"
HOOP_SEGMENT = MODEL_DIR / "hoop-segment.stl"
HOOP_COLLISION_SEGMENTS = {
    "mesh_path": HOOP_SEGMENT,
    "radius": 68.0 / 2.0,  # mm ring diameter -> 34 mm radius before scaling
    "yaw_step_deg": 29.9,
    "count": 12,
}


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
    """Spawn a single hoop. Parameters are left permissive so callers can
    place multiple hoops at arbitrary positions and with different appearance.
    """
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
        # Tweak material properties to your liking:
        lateral_friction=1.0,
        rolling_friction=0.05,
        spinning_friction=0.05,
        restitution=0.0,
    )
    print(f"[Hoop] Spawned hoop at {position} with orientation {orientation_euler_deg}.")


def add_hoops(arm: QArmBase, definitions: Iterable[dict]) -> None:
    """Spawn multiple hoops. Each definition is passed as kwargs to `add_hoop`.

    Example:
        defs = [
            {"position": (0, -0.3, 0.08)},
            {"position": (0.2, -0.25, 0.1), "rgba": (1, 0, 0, 1)},
        ]
        add_hoops(arm, defs)
    """
    # Respect optional 'enabled' flag on each definition
    enabled_defs = [d for d in definitions if d.get("enabled", True)]
    skipped = len(definitions) - len(enabled_defs)
    if skipped:
        print(f"[Hoop] Skipping {skipped} disabled hoop definition(s).")

    for d in enabled_defs:
        try:
            # Do not forward the internal 'enabled' flag to add_hoop
            kwargs = dict(d)
            kwargs.pop("enabled", None)
            add_hoop(arm, **kwargs)
        except TypeError as e:
            print(f"[Hoop] Invalid hoop definition {d}: {e}")


def run_viewer(arm: QArmBase) -> None:
    env = getattr(arm, "env", None)
    if env is None:
        print("[Hoop] No env attached; viewer unavailable.")
        return

    from sim.panda_viewer import PandaArmViewer, PhysicsBridge

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


def main() -> None:
    auto_step = not USE_PANDA_VIEWER
    arm = make_qarm(
        mode=MODE,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        time_step=TIME_STEP,
        auto_step=auto_step,
    )
    arm.home()
    if SPAWN_MULTIPLE_HOOPS:
        add_hoops(arm, DEFAULT_HOOP_DEFS)
    else:
        add_hoop(arm)

    try:
        if USE_PANDA_VIEWER:
            run_viewer(arm)
        else:
            print("[Hoop] Hoop loaded. Press Ctrl+C to quit.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Hoop] Exiting hoop demo.")
    finally:
        try:
            arm.home()
        except Exception:
            pass


if __name__ == "__main__":
    main()
