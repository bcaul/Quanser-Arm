#!/usr/bin/env python
"""
Quick launcher for the hoop sorting demo with Panda3D viewer.

Run from the workspace root:
    python run_hoop_sorting.py

This will:
  1. Load 16 hoop coordinates from demos/hoop_positions.json
  2. Spawn them in the simulation
  3. Open the Panda3D viewer
  4. Autonomously pick each hoop and place it on a pole
  5. Display everything in real-time

Close the viewer window to exit.
"""

from pathlib import Path
import sys

# Ensure we're in the right directory
workspace_root = Path(__file__).resolve().parent
if not (workspace_root / "api").exists():
    print(f"Error: This script must be run from the Quanser-Arm workspace root.")
    print(f"Current directory: {workspace_root}")
    sys.exit(1)

if __name__ == "__main__":
    from demos.hoop_sorter import main
    main()
