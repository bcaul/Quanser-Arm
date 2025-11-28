"""
Label/annotation demo for the QArm simulator.

Shows how to create and update custom point labels that appear in the Panda3D
viewer (red crosshairs with text). Use this as a reference for tagging targets,
waypoints, or regions of interest in your own projects.

Run it:
    python -m demos.label_demo
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

from api.factory import make_qarm
from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase

# --- quick toggles for students ---
MODE = "sim"  # keep on "sim" unless you have the real arm plus the hardware SDK
USE_PANDA_VIEWER = True  # labels show up in the Panda3D viewer
USE_PYBULLET_GUI = False
TIME_STEP = 1.0 / 240.0
HEADLESS_SECONDS = 6.0  # fallback spin time when no viewer is open
SHOW_VIEWER_SLIDERS = False
RELOAD_MESHES = False


def _launch_viewer(arm: QArmBase) -> None:
    """Start the Panda3D viewer so labels are visible."""
    env = getattr(arm, "env", None)
    if env is None:
        print("[Labels] No simulator attached; viewer cannot start.")
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


def _create_labels(env: object) -> dict[str, int]:
    """Add a few labels and return their ids for later updates."""
    if env is None or not hasattr(env, "add_point_label"):
        print("[Labels] Simulator not available; cannot create labels.")
        return {}
    add_label = env.add_point_label  # type: ignore[attr-defined]
    labels: dict[str, int] = {}
    labels["origin"] = add_label(
        "Base origin",
        (0.0, 0.0, 0.0),
        color=(0.2, 0.7, 1.0, 1.0),
        marker_scale=0.12,
        text_scale=0.03,
        show_coords=True,
    )
    labels["drop_zone"] = add_label(
        "Drop zone",
        (0.3, -0.25, 0.05),
        color=(0.9, 0.4, 0.2, 1.0),
        marker_scale=0.1,
        text_scale=0.025,
        show_coords=False,
    )
    labels["moving"] = add_label(
        "Moving waypoint",
        (0.2, 0.0, 0.1),
        color=(0.4, 0.95, 0.4, 1.0),
        marker_scale=0.12,
        text_scale=0.03,
        show_coords=True,
    )
    print(f"[Labels] Created {len(labels)} labels. Joint order: {DEFAULT_JOINT_ORDER}")
    return labels


def _orbit_label(env: object, label_id: int, stop_event: threading.Event) -> None:
    """Animate a label around the base in a gentle circle."""
    if env is None or not hasattr(env, "update_point_label"):
        return
    update_label = env.update_point_label  # type: ignore[attr-defined]
    radius = 0.22
    angle = 0.0
    while not stop_event.is_set():
        x = radius * math.cos(angle)
        y = -0.3 + radius * math.sin(angle)
        z = 0.1 + 0.05 * math.sin(angle * 0.5)
        update_label(label_id, position=(x, y, z), name="Moving waypoint")
        angle += 0.5 * TIME_STEP * 60.0  # moderate angular speed
        time.sleep(TIME_STEP)


def _headless_spin(env: object) -> None:
    """Keep physics alive briefly when no viewer is open."""
    if not hasattr(env, "step") or not hasattr(env, "time_step"):
        print("[Labels] No environment to step. Enable the simulator first.")
        return
    steps = int(HEADLESS_SECONDS / env.time_step)
    print(f"[Labels] Stepping headless for {HEADLESS_SECONDS:.1f}s ({steps} steps)...")
    for _ in range(steps):
        env.step()
        time.sleep(env.time_step)
    print("[Labels] Done stepping; press Ctrl+C to exit.")


def main() -> None:
    auto_step = not USE_PANDA_VIEWER
    arm = make_qarm(
        mode=MODE,
        gui=USE_PYBULLET_GUI,
        real_time=False,
        time_step=TIME_STEP,
        auto_step=auto_step,
    )
    env = getattr(arm, "env", None)
    if env is not None:
        env.reset()

    labels = _create_labels(env)
    stop = threading.Event()
    mover = None
    moving_id = labels.get("moving")
    if moving_id is not None and env is not None:
        mover = threading.Thread(target=_orbit_label, args=(env, moving_id, stop), daemon=True)
        mover.start()

    try:
        if USE_PANDA_VIEWER:
            _launch_viewer(arm)
        elif env is not None:
            _headless_spin(env)
        else:
            print("[Labels] No simulator to run; check MODE.")
    except KeyboardInterrupt:
        print("\n[Labels] Stopping label demo.")
    finally:
        stop.set()
        if mover is not None:
            mover.join(timeout=1.0)
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
