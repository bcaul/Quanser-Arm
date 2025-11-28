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

from api.factory import make_qarm
from common.qarm_base import QArmBase

MODE = "sim"
USE_PANDA_VIEWER = True
USE_PYBULLET_GUI = False
TIME_STEP = 1.0 / 240.0
MODEL_DIR = Path(__file__).resolve().parent / "models"

HOOP_MESH = MODEL_DIR / "hoop.stl"
HOOP_SEGMENT = MODEL_DIR / "hoop-segment.stl"
HOOP_COLLISION_SEGMENTS = {
    "mesh_path": HOOP_SEGMENT,
    "radius": 68.0 / 2.0,  # mm ring diameter -> 34 mm radius before scaling
    "yaw_step_deg": 29.9,
    "count": 12,
}


def add_hoop(arm: QArmBase) -> None:
    env = getattr(arm, "env", None)
    if env is None or not hasattr(env, "add_kinematic_object"):
        print("[Hoop] Simulator backend not available; skipping hoop spawn.")
        return
    if not HOOP_MESH.exists() or not HOOP_SEGMENT.exists():
        print(f"[Hoop] Missing hoop meshes under {MODEL_DIR}.")
        return

    env.add_kinematic_object(
        mesh_path=HOOP_MESH,
        position=(0.0, -0.30, 0.08),
        orientation_euler_deg=(0.0, 0.0, 20.0),
        scale=0.001,
        collision_segments=HOOP_COLLISION_SEGMENTS,
        mass=0.1,
        force_convex_for_dynamic=True,
        rgba=(0.1, 0.9, 0.1, 1.0),
        # Tweak material properties to your liking:
        lateral_friction=1.0,
        rolling_friction=0.05,
        spinning_friction=0.05,
        restitution=0.0,
    )
    print("[Hoop] Spawned hoop with collision segments. Open the viewer to inspect it.")


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
