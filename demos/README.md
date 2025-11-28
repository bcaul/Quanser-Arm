# Demo Pack

This folder holds short, beginner-friendly demos (each under ~200 lines) that show how
to drive the QArm in simulation or hardware. Open any file to see the constants you
can tweak: viewer toggles, joint waypoints, or small helper functions.

- `student_main.py` - quickstart: home + a few joint waypoints + open/close gripper. Panda3D viewer opens by default so you can watch the motion.
- `pick_and_place.py` - scripted sequence built from a small `POSES` dictionary so you can edit joint targets without touching logic. Panda3D viewer opens by default.
- `keyboard_control.py` - tiny REPL: tap keys to nudge joints, open/close the gripper, or return to home. Panda3D viewer opens by default.
- `blank_sim.py` - barebones "hello sim": launch the simulator + viewer with minimal code.
- `scene_objects.py` - drops the dog, head, and monkey meshes at preset offsets.
- `hoop_segments.py` - spawns a single hoop that uses collision segments for a more realistic ring shape.
- `gamepad_multi_hoops.py` - gamepad teleop (sticks + bumpers) with several collision-segmented hoops.
- `label_demo.py` - shows how to add and animate your own point labels in the viewer.

Run any demo with:
```bash
python -m demos.<module_name>
```
Leave `MODE = "sim"` for the simulator, or flip it to `"hardware"` once the real arm
is connected. Enable the PyBullet GUI by setting `USE_PYBULLET_GUI = True` inside a
script if you want the debug sliders.
