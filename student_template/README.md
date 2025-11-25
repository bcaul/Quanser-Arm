# Student Template

Use this folder as your safe sandbox to learn the QArm API. The entry script drives the
arm in joint space so you can plug in your own kinematics later.

## Fast path (see something move)

```bash
python -m student_template.student_main
```

This launches the simulation and opens the Panda3D viewer (Panda3D comes from `pip install -e .`).
Close the viewer window or press Ctrl+C to stop.

## What you just ran

- Creates a QArm in simulation mode via `api.make_qarm()`.
- Cycles through a handful of joint-space waypoints in the order `(yaw, shoulder, elbow, wrist)`.
- Keeps looping until the default duration expires.

## API

- `get_joint_positions()` → list of joint angles `(yaw, shoulder, elbow, wrist)`
- `set_joint_positions(q)` → command those joints
- `home()` → return to a safe pose

Feel free to copy/extend `student_main.py` and drop in your own FK/IK while keeping this API.

## Why is there an `__init__.py` here?

It marks this folder as a Python package so `python -m student_template.student_main` works
and the template can be imported cleanly elsewhere if needed.

## Extra info

- Run longer or forever: `python -m student_template.student_main --duration 0`
- Hide the viewer: `python -m student_template.student_main --headless`
- Debug with PyBullet sliders/GUI: `python -m student_template.student_main --pybullet-gui`
