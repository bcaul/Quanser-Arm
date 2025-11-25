"""Launch the full QArm simulation using PyBullet's built-in GUI controls."""

from __future__ import annotations

import argparse
import time

from sim.env import QArmSimEnv
from sim.assets import DEFAULT_BASE_ASSETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the main QArm sim with PyBullet debug sliders.")
    parser.add_argument("--real-time", action="store_true", help="Let PyBullet step in real time.")
    parser.add_argument("--time-step", type=float, default=1.0 / 120.0, help="Physics timestep (seconds).")
    parser.add_argument("--light-mode", action="store_true", help="Use a light PyBullet background.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_assets = DEFAULT_BASE_ASSETS

    env = QArmSimEnv(
        gui=True,
        real_time=args.real_time,
        time_step=args.time_step,
        dark_mode=not args.light_mode,
        enable_joint_sliders=True,
        show_debug_gui=True,
        add_ground=True,
        use_mesh_floor=True,
        base_assets=base_assets,
    )
    env.reset()

    print("Running with PyBullet debug sliders. Close the PyBullet window or Ctrl+C to exit.")
    try:
        while True:
            if env.enable_joint_sliders:
                env.apply_joint_slider_targets()
            if not args.real_time:
                env.step()
            time.sleep(env.time_step)
    except KeyboardInterrupt:
        pass
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
