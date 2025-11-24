"""
Headless PyBullet viewer that streams camera frames into a Qt window.

Run with:
    python -m sim.qt_viewer

This keeps PyBullet in DIRECT mode (no native GUI) and uses getCameraImage to
pull frames for display.
"""

from __future__ import annotations

import sys
import time

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("PyQt5 is required for the Qt viewer. Install with `pip install PyQt5`.") from exc

try:
    import pybullet as p
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit("pybullet is required for the Qt viewer. Install with `pip install -e .`.") from exc

def _apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Force a dark UI palette."""
    palette = QtGui.QPalette()
    dark_color = QtGui.QColor(45, 45, 48)
    disabled_color = QtGui.QColor(127, 127, 127)

    palette.setColor(QtGui.QPalette.Window, dark_color)
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.AlternateBase, dark_color)
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, dark_color)
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(0, 122, 204))
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)

    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.Text, disabled_color)
    palette.setColor(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, disabled_color)

    app.setStyle("Fusion")
    app.setPalette(palette)


def frange(start: float, stop: float, step: float):
    val = start
    while val <= stop + 1e-9:
        yield round(val, 6)
        val += step


from sim.env import QArmSimEnv


class CameraWindow(QtWidgets.QWidget):
    """Simple Qt window that displays frames from a PyBullet camera."""

    def __init__(self, env: QArmSimEnv, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.env = env
        self.label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(320, 240)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._last_frame: bytes | None = None
        self._grid_ids: list[int] = []

        # Camera state
        self._yaw = 45.0
        self._pitch = -30.0
        self._dist = 0.5
        self._target = [0.0, 0.0, 0.05]
        self._cam_base = (320, 240)  # lower base resolution for performance
        self._cam_scale = 0.5  # 50% render scale for performance

        # Build UI
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self._controls())
        self.setLayout(layout)

        self._add_grid_overlay()

        # Decouple simulation and rendering timers to keep the UI responsive.
        self._sim_timer = QtCore.QTimer(self)
        self._sim_timer.timeout.connect(self.env.step)
        self._sim_timer.start(max(int(self.env.time_step * 1000), 5))

        self._cam_timer = QtCore.QTimer(self)
        self._cam_timer.timeout.connect(self._update_frame)
        self._cam_timer.start(50)  # ~20 FPS capture

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # pragma: no cover - GUI hook
        for t in (getattr(self, "_sim_timer", None), getattr(self, "_cam_timer", None)):
            if t and t.isActive():
                t.stop()
        if self._grid_ids:
            for gid in self._grid_ids:
                try:
                    p.removeUserDebugItem(gid, physicsClientId=self.env.client)
                except Exception:
                    pass
        try:
            self.env.disconnect()
        except Exception:
            pass
        super().closeEvent(event)

    def _update_frame(self) -> None:
        # Grab a frame.
        width = int(self._cam_base[0] * self._cam_scale)
        height = int(self._cam_base[1] * self._cam_scale)
        w, h, rgba, depth = self.env.get_camera_image(
            width=width,
            height=height,
            yaw=self._yaw,
            pitch=self._pitch,
            distance=self._dist,
            target=self._target,
            return_depth=True,
        )
        # Replace background (depth ~1.0) with black.
        buf = bytearray(rgba)
        for i in range(len(depth)):
            if depth[i] >= 0.999:
                base = i * 4
                buf[base] = buf[base + 1] = buf[base + 2] = 0

        self._last_frame = bytes(buf)  # keep buffer alive for QImage
        image = QtGui.QImage(self._last_frame, w, h, QtGui.QImage.Format_RGBA8888)

        pixmap = QtGui.QPixmap.fromImage(image)
        scaled = pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.label.setPixmap(scaled)

    def _controls(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Camera Controls")
        grid = QtWidgets.QGridLayout()

        # Sliders for yaw/pitch/distance
        self._yaw_slider = self._make_slider(-180, 180, int(self._yaw), self._on_camera_change)
        self._pitch_slider = self._make_slider(-89, 89, int(self._pitch), self._on_camera_change)
        self._dist_slider = self._make_slider(10, 200, int(self._dist * 100), self._on_camera_change)
        self._scale_slider = self._make_slider(25, 100, int(self._cam_scale * 100), self._on_camera_change)

        grid.addWidget(QtWidgets.QLabel("Yaw (deg)"), 0, 0)
        grid.addWidget(self._yaw_slider, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Pitch (deg)"), 1, 0)
        grid.addWidget(self._pitch_slider, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Distance (m)"), 2, 0)
        grid.addWidget(self._dist_slider, 2, 1)
        grid.addWidget(QtWidgets.QLabel("Render scale (%)"), 3, 0)
        grid.addWidget(self._scale_slider, 3, 1)

        group.setLayout(grid)
        return group

    @staticmethod
    def _make_slider(min_val: int, max_val: int, start: int, callback) -> QtWidgets.QSlider:
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(start)
        slider.valueChanged.connect(callback)
        return slider

    def _on_camera_change(self) -> None:
        self._yaw = float(self._yaw_slider.value())
        self._pitch = float(self._pitch_slider.value())
        self._dist = max(0.05, self._dist_slider.value() / 100.0)
        self._cam_scale = max(0.2, self._scale_slider.value() / 100.0)

    def _add_grid_overlay(self) -> None:
        """Add a non-collidable grid using debug lines."""
        self._grid_ids.clear()
        size = 0.6
        step = 0.1  # fewer lines for performance
        color = (0.35, 0.35, 0.35)
        z = 0.0
        for x in frange(-size, size + 1e-6, step):
            gid = p.addUserDebugLine(
                lineFromXYZ=[x, -size, z],
                lineToXYZ=[x, size, z],
                lineColorRGB=color,
                lineWidth=1.0,
                physicsClientId=self.env.client,
            )
            self._grid_ids.append(gid)
        for y in frange(-size, size + 1e-6, step):
            gid = p.addUserDebugLine(
                lineFromXYZ=[-size, y, z],
                lineToXYZ=[size, y, z],
                lineColorRGB=color,
                lineWidth=1.0,
                physicsClientId=self.env.client,
            )
            self._grid_ids.append(gid)


def main() -> None:
    env = QArmSimEnv(gui=False, add_ground=False, enable_joint_sliders=False)
    env.reset()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    _apply_dark_palette(app)
    win = CameraWindow(env)
    win.setWindowTitle("QArm PyBullet (Qt viewer)")
    win.resize(960, 720)
    win.show()

    try:
        sys.exit(app.exec_())
    finally:
        try:
            env.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
