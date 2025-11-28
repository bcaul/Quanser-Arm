"""
Minimal QArm simulator bootstrap.

Use this as the cleanest starting point: connect to the simulator, keep it
alive, and drop in your own motion/kinematics code where noted.

Run it:
    python -m demos.blank_sim
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase

# --- quick toggles for students ---
MODE = "sim"  # leave as "sim" for the hackathon; switch to "hardware" later
USE_PANDA_VIEWER = True  # Panda3D viewer (preferred way to see the arm)
USE_PYBULLET_GUI = False  # set True to debug with Bullet's sliders/debug UI
TIME_STEP = 1.0 / 240.0
HEADLESS_SECONDS = 10.0  # how long to run when no viewer is open
SHOW_VIEWER_SLIDERS = False
RELOAD_MESHES = False


def _launch_viewer(arm: QArmBase) -> None:
    """Spin up the Panda3D viewer that renders the PyBullet simulation."""
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
    """Step physics without rendering—handy if you only need the backend running."""
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
    # Auto-step is disabled when the viewer runs because the viewer drives stepping.
    auto_step = not USE_PANDA_VIEWER
    print(f"[Blank] Connecting to QArm in {MODE} mode with joint order {DEFAULT_JOINT_ORDER}")
    arm = make_qarm(
        mode=MODE,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        time_step=TIME_STEP,
        auto_step=auto_step,
    )

    env = getattr(arm, "env", None)
    if env is not None:
        env.reset()  # zero pose; change this if you want a different starting stance

    try:
        if USE_PANDA_VIEWER:
            print("[Blank] Viewer opening. Add your own joint commands below this line.")
            _launch_viewer(arm)
        elif env is not None:
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


if __name__ == "__main__":
    main()
