"""
Scene helper demo: drop the dog, head, and monkey meshes.

- Meshes come from `demos/models/`.
- Placements reuse the first three hoop offsets from the earlier hoop demos.
- Keeps the code short so you can copy/paste the parts you need.

Run it:
    python -m demos.scene_objects
Turn off `USE_PANDA_VIEWER` if you only want the PyBullet GUI.
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

OBJECT_POSITIONS = [
    (0.0, -0.30, 0.08),
    (0.1, -0.35, 0.08),
    (-0.1, -0.40, 0.08),
]


def add_objects_and_labels(arm: QArmBase) -> None:
    env = getattr(arm, "env", None)
    if env is None or not hasattr(env, "add_kinematic_object"):
        print("[Scene] Simulator backend not available; skipping object spawn.")
        return

    meshes = {
        "monkey": MODEL_DIR / "blender_monkey.stl",
        "dog": MODEL_DIR / "dog.STL",
        "head": MODEL_DIR / "head.stl",
    }
    for name, path in meshes.items():
        if not path.exists():
            print(f"[Scene] Missing mesh at {path}")
            return

    placements = [
        {
            "mesh_path": meshes["dog"],
            "position": OBJECT_POSITIONS[0],
            "orientation_euler_deg": (0.0, 0.0, 0.0),
            "scale": 0.001,
            "mass": 0.5,
            "force_convex_for_dynamic": True,
            "rgba": (0.25, 0.5, 0.95, 1.0),
        },
        {
            "mesh_path": meshes["head"],
            "position": OBJECT_POSITIONS[1],
            "orientation_euler_deg": (0.0, 0.0, 20.0),
            "scale": 0.003,
            "mass": 0.5,
            "force_convex_for_dynamic": True,
            "rgba": (0.95, 0.8, 0.2, 1.0),
        },
        {
            "mesh_path": meshes["monkey"],
            "position": OBJECT_POSITIONS[2],
            "orientation_euler_deg": (0.0, 0.0, -15.0),
            "scale": 0.05,
            "mass": 0.5,
            "force_convex_for_dynamic": True,
            "rgba": (0.85, 0.25, 0.25, 1.0),
        },
    ]

    for obj in placements:
        env.add_kinematic_object(**obj)
    print("[Scene] Spawned dog, head, and monkey meshes. Open the viewer to inspect them.")


def run_viewer(arm: QArmBase) -> None:
    env = getattr(arm, "env", None)
    if env is None:
        print("[Scene] No env attached; viewer unavailable.")
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
    add_objects_and_labels(arm)

    try:
        if USE_PANDA_VIEWER:
            run_viewer(arm)
        else:
            print("[Scene] Objects loaded. Press Ctrl+C to quit.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[Scene] Exiting scene demo.")
    finally:
        try:
            arm.home()
        except Exception:
            pass


if __name__ == "__main__":
    main()
