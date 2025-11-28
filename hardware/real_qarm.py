"""
Hardware-based QArm controller that will wrap the Quanser Python API.

Quanser-specific imports and calls are intentionally omitted; fill them in when
the hardware SDK is available in the hackathon environment.
"""

from __future__ import annotations

from typing import Any, Sequence

from common.qarm_base import DEFAULT_JOINT_ORDER, QArmBase


class RealQArm(QArmBase):
    """Hardware QArm wrapper.

    This implementation attempts to use the vendor `QArm` API (as used in
    `hardware/BasicIO_position_mode_joint_space.py`) but keeps imports and calls
    guarded so the package remains importable on machines that do not have the
    vendor SDK installed.

    Behavior:
    - If a `client` object is provided it will be used directly.
    - Otherwise the constructor will try to import and instantiate
      `pal.products.qarm.QArm` (entering its context) and keep it as the
      active client.
    - If no client is available, methods raise NotImplementedError with an
      explanatory message.
    """

    OPEN_GRIP = 0.0
    CLOSED_GRIP = 1.0

    def __init__(self, client: Any | None = None) -> None:
        self.joint_name_hint: tuple[str, ...] = DEFAULT_JOINT_ORDER
        self._last_joint_positions: list[float] | None = None
        self._last_gripper: float | None = None

        if client is not None:
            # User-supplied client (useful for testing/mocking)
            self.client = client
            self._owns_client = False
            return

        # Try to import the vendor API and create a persistent client. Keep
        # import-time failures quiet so the package can be imported on systems
        # without the SDK installed.
        try:
            from pal.products.qarm import QArm  # type: ignore

            # The QArm object used in examples is a context manager. Create an
            # instance and enter its context so we keep an open connection.
            _qarm = QArm(hardware=1, readMode=0)
            try:
                self.client = _qarm.__enter__()
            except Exception:
                # If __enter__ fails, fall back to storing the raw object and
                # let calls fail later with a helpful message.
                self.client = _qarm

            self._owns_client = True
        except Exception:
            self.client = None
            self._owns_client = False

    def _require_client(self, call: str) -> Any:
        if self.client is None:
            raise NotImplementedError(
                f"RealQArm.{call} requires the Quanser hardware SDK and a connected robot. "
                "Use mode='sim' during the hackathon, or supply a hardware client when available."
            )
        return self.client

    def home(self) -> None:
        client = self._require_client("home")
        # Prefer an explicit home() if provided by the SDK, otherwise move to
        # a sensible zero joint pose.
        if hasattr(client, "home"):
            client.home()
            return

        zero_q = [0.0, 0.0, 0.0, 0.0]
        self.set_joint_positions(zero_q)

    def set_joint_positions(self, q: Sequence[float]) -> None:
        client = self._require_client("set_joint_positions")
        q_list = list(map(float, q))
        self._last_joint_positions = q_list

        # Many example scripts call `read_write_std(phiCMD=..., gprCMD=...)`.
        # Prefer that method if present.
        if hasattr(client, "read_write_std"):
            gpr = self._last_gripper if self._last_gripper is not None else 0.0
            try:
                client.read_write_std(phiCMD=q_list, gprCMD=gpr)
                return
            except TypeError:
                # Some SDKs may use different arg names/order; fall through.
                pass

        # Fallback: try a generic command method if available.
        if hasattr(client, "set_joint_positions"):
            client.set_joint_positions(q_list)
            return

        raise NotImplementedError("Underlying QArm client does not support sending joint positions")

    def get_joint_positions(self) -> list[float]:
        client = self._require_client("get_joint_positions")

        # If the SDK exposes a read/state API, prefer that.
        if hasattr(client, "get_joint_positions"):
            return list(map(float, client.get_joint_positions()))

        if hasattr(client, "read_state"):
            state = client.read_state()
            # best-effort: extract a positions field if present
            if hasattr(state, "positions"):
                return list(map(float, state.positions))

        if self._last_joint_positions is not None:
            return list(self._last_joint_positions)

        raise NotImplementedError("Unable to read joint positions from underlying client")

    def set_gripper_position(self, angle: float) -> None:
        self.set_gripper_positions([angle])

    def set_gripper_positions(self, angles: Sequence[float]) -> None:
        client = self._require_client("set_gripper_positions")
        # Use a single scalar if the SDK expects one value
        g = float(angles[0]) if len(angles) else 0.0
        self._last_gripper = g

        if hasattr(client, "read_write_std"):
            q = self._last_joint_positions if self._last_joint_positions is not None else [0.0, 0.0, 0.0, 0.0]
            try:
                client.read_write_std(phiCMD=q, gprCMD=g)
                return
            except TypeError:
                pass

        if hasattr(client, "set_gripper_position"):
            client.set_gripper_position(g)
            return

        raise NotImplementedError("Underlying QArm client does not support setting gripper position")

    def open_gripper(self) -> None:
        self.set_gripper_position(self.OPEN_GRIP)

    def close_gripper(self) -> None:
        self.set_gripper_position(self.CLOSED_GRIP)

    def terminate(self) -> None:
        # Clean up the owned client if possible.
        if getattr(self, "_owns_client", False) and self.client is not None:
            try:
                # If client was created via context manager, call __exit__.
                if hasattr(self.client, "__exit__"):
                    # __exit__ accepts exc_type, exc, tb; pass None for normal close
                    self.client.__exit__(None, None, None)
                elif hasattr(self.client, "terminate"):
                    self.client.terminate()
            except Exception:
                # Swallow termination errors to avoid raising on interpreter exit.
                pass

