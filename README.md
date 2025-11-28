# QArm Hackathon Framework (WIP)

This repository provides the core structure for a hackathon centred around the
Quanser QArm robotic manipulator.

You will control a QArm in simulation (and later on real hardware) by
commanding joint angles and implementing their own kinematics and strategies
for picking and placing coloured hoops onto stands.

## Current status

- `sim/qarm/` contains the URDF and mesh resources for the QArm.
- `common.QArmBase` defines the joint-space API and default joint ordering.
- `sim.env.QArmSimEnv` runs the PyBullet backend; `sim.SimQArm` wraps it to match `QArmBase`.
- `api.make_qarm` switches between simulation (default, headless) and the stubbed `hardware.RealQArm`.
- `demos/` holds simplified student demos (quickstart, pick/place, keyboard control, gamepad hoops, scene helpers).
- `hardware.RealQArm` remains a stub until the Quanser SDK is available.

## Local setup (for devs and students)

```bash
cd /qarm-hack      # or clone destination
python3 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                            # installs pybullet + numpy
```

Quick smoke test (no GUI):
```bash
python - <<'PY'
import pybullet as p
cid = p.connect(p.DIRECT)
robot = p.loadURDF("sim/qarm/urdf/QARM.urdf")
print("client:", cid, "robot:", robot)
p.disconnect()
PY
```

- On macOS, you may need Xcode Command Line Tools once: `xcode-select --install`.
- PyBullet builds a native wheel; it can take a minute on first install.
- **Every new shell**: reactivate the venv (`source .venv/bin/activate`). If you forget,
  `python/pip` will fall back to the system interpreter and you may see PEP 668 errors
  about an "externally managed environment."

## New to Python/VSCode? Start here

- Install VSCode from code.visualstudio.com (Windows/macOS/Linux).
- Install Python 3.11+:
  - Windows: grab the installer from python.org, check "Add python.exe to PATH".
  - macOS: `brew install python` or use the python.org installer.
  - Ubuntu/Debian: `sudo apt-get install python3 python3-venv python3-pip`.
- Launch VSCode and install the "Python" extension (ms-python.python). Pylance is recommended too.
- Create/select the project venv in VSCode: `Ctrl+Shift+P` → start typing **Python: Create Environment** → select it → choose `Venv` and the Python you installed. VSCode will wire the workspace to that interpreter.
- Open a new VSCode terminal; it should auto-activate `.venv` (or run the `source .venv/bin/activate` / `.venv\Scripts\activate` command shown above).
- Run `pip install -e .` in that terminal to pull dependencies into the venv. If the interpreter shown in VSCode's status bar is not your venv, click it and pick the `.venv` interpreter.

- **Why bother with a venv?** It keeps this project's packages (pybullet, panda3d, etc.)
  isolated from your system Python so you don't pollute or break other projects, and you
  get a repeatable set of dependencies that matches your teammates'.

## Where to start coding

Open `demos/README.md` for the menu of short demos. The quickest path is
`demos/student_main.py`, which homes the arm, runs a few joint-space
waypoints, and optionally opens the Panda viewer. Other demos show a scripted
pick-and-place, keyboard nudging, and adding a hoop + labels to the scene. Keep the
joint order `(yaw, shoulder, elbow, wrist)` when you start dropping in your own
kinematics.

**Want to see something running?**
```bash
python -m demos.student_main
```
Run this ^ command in your terminal.

## Running the sims (Panda3D-first, PyBullet for debug)

- **Panda3D viewport (recommended for visuals):**
  ```bash
  python -m sim.panda_viewer
  ```
  (Requires `panda3d` installed; included in `pip install -e .`.)
- **PyBullet debug sliders / GUI (useful for quick joint pokes):**
  ```bash
python -m sim.actual_sim --real-time
python -m sim.run_gui --gui --real-time --sliders
```
- **Student demos (joint-space API):**
  ```bash
  python -m demos.student_main              # quickstart wave
  python -m demos.pick_and_place            # simple scripted pick/place
  python -m demos.keyboard_control          # manual nudges + gripper
  python -m demos.scene_objects             # meshes in the scene
  python -m demos.hoop_segments             # single hoop with collision segments
  python -m demos.gamepad_multi_hoops       # gamepad teleop with multiple hoops
  ```
  (Edit the constants at the top of each file to change duration, viewer, or GUI toggles.)

Base and accent meshes are now hardcoded in `sim/assets.py` (pinebase + collision and green/blue accents) with a shared 0.001 visual/collision scale, so PyBullet and Panda3D stay in sync without passing extra CLI arguments.

### VSCode run/debug helpers

- The repo tracks `.vscode/launch.json` so you get the launch targets on clone. The file contents:

```jsonc
{
  // Use IntelliSense to learn about possible attributes.
  // Hover to view descriptions of existing attributes.
  // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Student Sandbox (default)",
      "type": "debugpy",
      "request": "launch",
      "module": "demos.student_main",
      "justMyCode": true,
      "args": []
    },
    {
      "name": "Panda Viewer (Debug)",
      "type": "debugpy",
      "request": "launch",
      "module": "sim.panda_viewer",
      "justMyCode": true,
      "args": []
    },
    {
      "name": "PyBullet GUI (Debug)",
      "type": "debugpy",
      "request": "launch",
      "module": "sim.run_gui",
      "justMyCode": true,
      "args": ["--gui", "--sliders", "--real-time"]
    }
  ]
}
```

- The first entry launches the beginner sandbox and is the default in VSCode. Other configs open the Panda viewer or PyBullet GUI/debug sliders. Base meshes and scales are baked into the sim (see `sim/assets.py`), so no extra mesh arguments are required.
