"""
Run the qarm hoop sorter in the simulation with the Panda3D viewer.

This script:
 - creates a simulated QArm (PyBullet + Panda3D bridge)
 - spawns the 16 hoops from `demos/hoop_positions.json`
 - runs the sorting routine from `demos/qarm_hoop_sorter.py` in a background thread
 - launches the Panda3D viewer so you can watch the arm move

Usage:
    python demos/run_sorter_in_sim.py

"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from api.factory import make_qarm
from sim.panda_viewer import PhysicsBridge, PandaArmViewer

# Import sorter utilities
from demos.qarm_hoop_sorter import sort_all_hoops

ROOT = Path(__file__).resolve().parent
HOOP_JSON = ROOT / "hoop_positions.json"
HOOP_MESH = ROOT / "models" / "hoop.stl"


class SimArmAdapter:
    """Adapter exposing the minimal API expected by the sorter script.

    Maps:
      - get_joint_angles() -> get_joint_positions()
      - move_joint_angles(angles) -> set_joint_positions(angles)
      - control_gripper(close=True/False) -> set_gripper_position(angle)
    """

    def __init__(self, sim_arm: Any):
        self._arm = sim_arm

    def get_joint_angles(self):
        return list(self._arm.get_joint_positions())

    def move_joint_angles(self, angles):
        # Use the smooth, blocking set_joint_positions API so motion looks realistic
        try:
            self._arm.set_joint_positions(angles)
        except Exception:
            # fallback to instant
            self._arm.set_joint_positions_instant(angles)

    def control_gripper(self, close: bool = True):
        # Try to use meaningful open/close positions; fall back to 0.0/0.5
        try:
            closed = getattr(self._arm, "_closed_angle", 0.5)
            opened = getattr(self._arm, "_open_angle", 0.0)
            self._arm.set_gripper_position(closed if close else opened)
        except Exception:
            try:
                self._arm.set_gripper_position(0.5 if close else 0.0)
            except Exception:
                pass


def spawn_hoops(env, hoops):
    # spawn kinematic hoop meshes in the sim environment
    mesh_path = HOOP_MESH
    if not mesh_path.exists():
        print(f"[spawn_hoops] hoop mesh missing: {mesh_path}")
        return
    for i, spec in enumerate(hoops, start=1):
        pos = spec["position"]
        rgba = spec.get("rgba", (0.5, 0.5, 0.5, 1.0))
        try:
            env.add_kinematic_object(
                mesh_path=mesh_path,
                position=tuple(pos),
                orientation_euler_deg=(0.0, 0.0, 0.0),
                scale=0.001,
                collision_segments=None,
                mass=0.1,
                force_convex_for_dynamic=True,
                rgba=tuple(rgba),
            )
            print(f"[spawn_hoops] spawned hoop #{i} at {pos}")
        except Exception as e:
            print(f"[spawn_hoops] failed to spawn hoop #{i}: {e}")


def main():
    # create sim QArm (headless PyBullet; Panda viewer will show visuals)
    sim_arm = make_qarm(mode="sim", gui=False, real_time=False, time_step=1.0 / 240.0, auto_step=False)
    sim_arm.home()

    # load hoop positions
    hoops = []
    if HOOP_JSON.exists():
        with HOOP_JSON.open("r") as f:
            data = json.load(f)
        for obj in data:
            if isinstance(obj, dict) and "position" in obj:
                hoops.append({"position": obj["position"], "rgba": obj.get("rgba", [0.5, 0.5, 0.5, 1.0])})
    else:
        print(f"Hoop JSON not found: {HOOP_JSON}")

    # spawn hoops in the simulation environment
    spawn_hoops(sim_arm.env, hoops)

    # start sorter in background thread (uses the adapter)
    adapter = SimArmAdapter(sim_arm)

    def sorter_thread():
        try:
            # sort_all_hoops returns placed_count (or None); pass pole near center
            sort_all_hoops(adapter, str(HOOP_JSON), (0.0, -0.3, 0.15), elbow='elbow_down')
        except Exception as e:
            print("[sorter_thread] error:", e)

    t = threading.Thread(target=sorter_thread, daemon=True)
    t.start()

    # Launch Panda viewer (this blocks until user closes window)
    bridge = PhysicsBridge(time_step=sim_arm.env.time_step, env=sim_arm.env, reset=False)
    viewer_args = SimpleNamespace(
        time_step=sim_arm.env.time_step,
        hide_base=False,
        hide_accents=False,
        probe_base_collision=False,
        show_sliders=False,
        reload_meshes=False,
    )
    viewer = PandaArmViewer(bridge, viewer_args)
    try:
        viewer.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
