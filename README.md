# QArm Hackathon Framework (WIP)

This repository provides the core structure for a hackathon centred around the
Quanser QArm robotic manipulator.

Students will control a QArm in simulation (and later on real hardware) by
commanding joint angles and implementing their own kinematics and strategies
for picking and placing coloured hoops onto stands.

## Current status

- `sim/qarm/` contains the URDF and mesh resources for the QArm.
- `common.QArmBase` defines the joint-space API and default joint ordering.
- `sim.env.QArmSimEnv` runs the PyBullet backend; `sim.SimQArm` wraps it to match `QArmBase`.
- `api.make_qarm` switches between simulation (default, headless) and the stubbed `hardware.RealQArm`.
- `student_template/` holds a runnable joint-control sandbox for teams.
- `hardware.RealQArm` remains a stub until the Quanser SDK is available.

## Local setup (for devs and students)

```bash
cd /qarm-hack      # or clone destination
python3 -m venv .venv
source .venv/bin/activate                   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .                            # installs pybullet + numpy
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
- **Student sandbox (joint-space API, Panda3D by default):**
  ```bash
  python -m student_template.student_main                         # opens Panda3D viewport (default)
  python -m student_template.student_main --headless              # run without Panda3D
  python -m student_template.student_main --pybullet-gui          # debug GUI if needed
  python -m student_template.student_main --pybullet-gui --headless  # debug GUI only
  ```

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
      "name": "Actual Sim (PyBullet GUI)",
      "type": "debugpy",
      "request": "launch",
      "module": "sim.actual_sim",
      "justMyCode": true,
      "args": ["--real-time"]
    },
    {
      "name": "Quick Run (run_gui)",
      "type": "debugpy",
      "request": "launch",
      "module": "sim.run_gui",
      "justMyCode": true,
      "args": ["--gui", "--sliders", "--real-time"]
    },
    {
      "name": "Panda Viewer",
      "type": "debugpy",
      "request": "launch",
      "module": "sim.panda_viewer",
      "justMyCode": true,
      "args": [
        "--base-mesh",
        "sim/models/pinebase.stl",
        "--base-collision-mesh",
        "sim/models/pinebase_collision.stl",
        "--base-scale",
        "0.001",
        "--base-yaw",
        "180",
        "--base-friction",
        "0.8",
        "--base-restitution",
        "0.0",
        "--probe-base-collision"
      ]
    }
  ]
}
```

- Launch configs use PyBullet's own GUI and debug sliders. `--real-time` is pre-wired; add more args as needed.
