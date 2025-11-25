"""
Copy of student_main focused on dropping a kinematic mesh into the sim.

Run with:
    python -m student_template.student_object
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from api.factory import make_qarm
from common.qarm_base import QArmBase

# Viewer toggles match student_main defaults.
USE_PANDA_VIEWER = True
USE_PYBULLET_GUI = False

# Preload the bundled Blender monkey so students see a static mesh instantly.
KINEMATIC_OBJECTS: list[dict[str, object]] = [
    {
        "mesh_path": Path(__file__).parent / "blender_monkey.stl",
        "position": (0.0, -0.3, 0.08),
        "euler_deg": (0.0, 0.0, 45.0),  # roll, pitch, yaw in degrees
        "scale": 0.1,
        "mass": 0.5,  # tweak this to let gravity act on the mesh; set to 0 for static
        "force_convex_for_dynamic": True,  # keep True for stable collisions with the base
    }
]


def add_kinematic_objects(arm: QArmBase, objects: list[dict[str, object]]) -> None:
    """
    Convenience wrapper around the simulator's kinematic mesh helper.
    Pass a list of dicts shaped like KINEMATIC_OBJECTS above.
    """
    if not objects:
        return
    env = getattr(arm, "env", None)
    add_fn = getattr(arm, "add_kinematic_object", None)
    if add_fn is None and env is not None:
        add_fn = getattr(env, "add_kinematic_object", None)
    if add_fn is None:
        print("[Student] Current QArm backend does not support kinematic objects.")
        return
    for obj in objects:
        kwargs = {
            "mesh_path": obj["mesh_path"],
            "position": obj.get("position", (0.0, 0.0, 0.0)),
            "scale": obj.get("scale", 0.1),
            "rgba": obj.get("rgba"),
            "mass": obj.get("mass", 1.0),
            "force_convex_for_dynamic": obj.get("force_convex_for_dynamic", True),
        }
        if "quat_xyzw" in obj:
            kwargs["orientation_quat_xyzw"] = obj["quat_xyzw"]
        else:
            kwargs["orientation_euler_deg"] = obj.get("euler_deg")
        body_id = add_fn(**kwargs)
        print(
            "[Student] Added kinematic mesh",
            obj["mesh_path"],
            f"(scale={kwargs['scale']}, mass={kwargs['mass']}, convex_dynamic={kwargs['force_convex_for_dynamic']}, body_id={body_id})",
        )


def main() -> None:
    use_panda_viewer = USE_PANDA_VIEWER
    use_pybullet_gui = USE_PYBULLET_GUI

    # Keep PyBullet stepping driven by the Panda viewer.
    real_time = False
    auto_step = not use_panda_viewer

    arm = make_qarm(mode="sim", gui=use_pybullet_gui, real_time=real_time, auto_step=auto_step)
    arm.home()
    add_kinematic_objects(arm, KINEMATIC_OBJECTS)
    time.sleep(0.1)

    def launch_viewer() -> None:
        from sim.panda_viewer import PandaArmViewer, PhysicsBridge

        viewer_args = SimpleNamespace(
            time_step=arm.env.time_step,
            hide_base=False,
            hide_accents=False,
            probe_base_collision=False,
        )
        physics = PhysicsBridge(
            time_step=arm.env.time_step,
            env=getattr(arm, "env", None),
            reset=False,
        )
        app = PandaArmViewer(physics, viewer_args)
        app.run()

    try:
        if use_panda_viewer:
            launch_viewer()  # blocks until closed
        else:
            print("[Student] Kinematic objects loaded. Press Ctrl+C to exit.")
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nStopping kinematic object demo...")
    finally:
        try:
            arm.home()
        except Exception:
            pass


if __name__ == "__main__":
    main()
