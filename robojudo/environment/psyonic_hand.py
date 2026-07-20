"""Psyonic Ability Hand interface (left + right) for RoboJuDo.

Follows the repo's "real vs mock backend" pattern (cf. DummyEnv / Unitree envs).

This module imports cleanly WITHOUT any hardware attached and WITHOUT the
``ability-hand-api`` package installed. The real serial backend is lazily
imported only when a :class:`RealPsyonicBackend` is actually connected, so a
mock configuration can be exercised anywhere.

Finger index map used throughout (6 DoF, degrees)::

    [index, middle, ring, pinky, thumb_flexor, thumb_rotator]

The thumb rotator (last element) uses a negative range on real hardware.
"""

from __future__ import annotations

import abc
import logging
from typing import Literal

# Allow direct execution (`python robojudo/environment/psyonic_hand.py`), which
# otherwise puts only this file's dir on sys.path and hides the `robojudo` pkg.
if __package__ in (None, ""):
    import os as _os
    import sys as _sys

    _repo_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)

from robojudo.config import Config

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================
class PsyonicHandCfg(Config):
    enabled: bool = False
    backend: Literal["mock", "real"] = "mock"
    left_port: str | None = None  # e.g. "/dev/ttyUSB0"
    right_port: str | None = None  # e.g. "/dev/ttyUSB1"
    baud_rate: int = 460800
    reply_mode: int = 0  # 0: Pos.Cur.Touch  1: Pos.Vel.Touch  2: Pos.Cur.Vel
    rate_hz: int = 100
    num_fingers: int = 6
    # A safe partially-open ready pose (6 dof). Thumb rotator negative.
    ready_position: list[float] = [30.0, 30.0, 30.0, 30.0, 30.0, -30.0]
    # Path to ability-hand-api/python dir; prepended to sys.path in real backend.
    ability_hand_api_path: str | None = None


# ======================================================================
# Backend abstract base
# ======================================================================
class PsyonicHandBackend(abc.ABC):
    """One physical (or simulated) hand."""

    def __init__(self, side: str, cfg: PsyonicHandCfg):
        assert side in ("left", "right"), f"invalid side: {side}"
        self.side = side
        self.cfg = cfg

    @abc.abstractmethod
    def connect(self) -> None: ...

    @abc.abstractmethod
    def set_position(self, positions: list[float]) -> None: ...

    @abc.abstractmethod
    def set_velocity(self, velocities: list[float]) -> None: ...

    @abc.abstractmethod
    def set_torque(self, currents: list[float]) -> None: ...

    @abc.abstractmethod
    def set_grip(self, cmd: int, speed: int = 0xFF) -> None: ...

    @abc.abstractmethod
    def get_position(self) -> list[float] | None: ...

    @abc.abstractmethod
    def get_velocity(self) -> list[float] | None: ...

    @abc.abstractmethod
    def get_current(self) -> list[float] | None: ...

    @abc.abstractmethod
    def get_touch(self) -> list[float] | None:
        """Alias to the FSR / touch sensors."""
        ...

    @abc.abstractmethod
    def close(self) -> None: ...


# ======================================================================
# Mock backend
# ======================================================================
class MockPsyonicBackend(PsyonicHandBackend):
    """Hardware-free deterministic backend.

    Holds a commanded target and a simulated current position. Each read
    advances the simulated position toward the last commanded target with a
    simple first-order lag. No hardware, no sleeps in the read path.
    """

    LAG = 0.3

    def __init__(self, side: str, cfg: PsyonicHandCfg):
        super().__init__(side, cfg)
        n = cfg.num_fingers
        ready = list(cfg.ready_position)
        # Guard against a mismatched ready_position length.
        if len(ready) != n:
            ready = (ready + [0.0] * n)[:n]
        self._target: list[float] = list(ready)
        self._pos: list[float] = list(ready)
        self._last_delta: list[float] = [0.0] * n
        self._connected = False

    def connect(self) -> None:
        self._connected = True
        logger.info("[Psyonic:%s:mock] connected", self.side)

    def _require(self) -> None:
        if not self._connected:
            raise RuntimeError(
                f"[Psyonic:{self.side}:mock] not connected; call connect() first"
            )

    def set_position(self, positions: list[float]) -> None:
        self._require()
        self._target = list(positions)
        logger.debug("[Psyonic:%s:mock] set_position %s", self.side, positions)

    def set_velocity(self, velocities: list[float]) -> None:
        self._require()
        logger.debug("[Psyonic:%s:mock] set_velocity %s", self.side, velocities)

    def set_torque(self, currents: list[float]) -> None:
        self._require()
        logger.debug("[Psyonic:%s:mock] set_torque %s", self.side, currents)

    def set_grip(self, cmd: int, speed: int = 0xFF) -> None:
        self._require()
        logger.debug("[Psyonic:%s:mock] set_grip cmd=%s speed=%s", self.side, cmd, speed)

    def _advance(self) -> None:
        new_pos = []
        delta = []
        for p, t in zip(self._pos, self._target):
            d = self.LAG * (t - p)
            new_pos.append(p + d)
            delta.append(d)
        self._pos = new_pos
        self._last_delta = delta

    def get_position(self) -> list[float] | None:
        self._require()
        self._advance()
        return list(self._pos)

    def get_velocity(self) -> list[float] | None:
        self._require()
        return list(self._last_delta)

    def get_current(self) -> list[float] | None:
        self._require()
        return [0.0] * self.cfg.num_fingers

    def get_touch(self) -> list[float] | None:
        self._require()
        return [0.0] * self.cfg.num_fingers

    def close(self) -> None:
        self._connected = False
        logger.info("[Psyonic:%s:mock] closed", self.side)


# ======================================================================
# Real backend
# ======================================================================
class RealPsyonicBackend(PsyonicHandBackend):
    """Serial-backed Ability Hand. Requires hardware + ability-hand-api.

    The real API is imported lazily inside :meth:`connect` so this module (and
    a mock configuration) load with neither the package nor a device present.
    """

    def __init__(self, side: str, cfg: PsyonicHandCfg):
        super().__init__(side, cfg)
        self._client = None
        self._port = cfg.left_port if side == "left" else cfg.right_port

    def connect(self) -> None:
        if not self._port:
            raise RuntimeError(
                f"[Psyonic:{self.side}:real] no port configured for this hand"
            )
        try:
            if self.cfg.ability_hand_api_path:
                import sys

                if self.cfg.ability_hand_api_path not in sys.path:
                    sys.path.insert(0, self.cfg.ability_hand_api_path)
            from ah_wrapper.ah_serial_client import AHSerialClient
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "[Psyonic:%s:real] could not import ability-hand-api "
                "(ah_wrapper.ah_serial_client). Real hardware backend requires "
                "the ability-hand-api package on sys.path; set "
                "PsyonicHandCfg.ability_hand_api_path or install it. "
                "Underlying error: %r" % (self.side, e)
            ) from e

        try:
            self._client = AHSerialClient(
                port=self._port,
                baud_rate=self.cfg.baud_rate,
                reply_mode=self.cfg.reply_mode,
                rate_hz=self.cfg.rate_hz,
            )
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "[Psyonic:%s:real] failed to open Ability Hand on port %r. "
                "Hardware must be connected and permitted. Underlying error: %r"
                % (self.side, self._port, e)
            ) from e
        logger.info("[Psyonic:%s:real] connected on %s", self.side, self._port)

    def _require(self):
        if self._client is None:
            raise RuntimeError(
                f"[Psyonic:{self.side}:real] not connected; call connect() first"
            )
        return self._client

    def set_position(self, positions: list[float]) -> None:
        self._require().set_position(list(positions))

    def set_velocity(self, velocities: list[float]) -> None:
        self._require().set_velocity(list(velocities))

    def set_torque(self, currents: list[float]) -> None:
        self._require().set_torque(list(currents))

    def set_grip(self, cmd: int, speed: int = 0xFF) -> None:
        self._require().set_grip(cmd, speed=speed)

    def get_position(self) -> list[float] | None:
        return self._require().hand.get_position()

    def get_velocity(self) -> list[float] | None:
        return self._require().hand.get_velocity()

    def get_current(self) -> list[float] | None:
        return self._require().hand.get_current()

    def get_touch(self) -> list[float] | None:
        # FSR == touch sensors.
        return self._require().hand.get_fsr()

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
            logger.info("[Psyonic:%s:real] closed", self.side)


# ======================================================================
# Manager
# ======================================================================
_BACKENDS = {"mock": MockPsyonicBackend, "real": RealPsyonicBackend}


class PsyonicHands:
    """Manage the left + right Ability Hands.

    A hand is "present" only if its port is configured. A hand with a ``None``
    port is skipped, so single-hand setups work transparently.
    """

    def __init__(self, cfg: PsyonicHandCfg):
        self.cfg = cfg
        backend_cls = _BACKENDS[cfg.backend]
        self._backends: dict[str, PsyonicHandBackend] = {}
        for side, port in (("left", cfg.left_port), ("right", cfg.right_port)):
            if port is not None:
                self._backends[side] = backend_cls(side, cfg)
        if not self._backends:
            logger.warning(
                "[PsyonicHands] no ports configured (left_port/right_port both None); "
                "no hands will be managed"
            )

    @property
    def present_sides(self) -> list[str]:
        return list(self._backends.keys())

    def connect(self) -> None:
        for side, backend in self._backends.items():
            logger.info("[PsyonicHands] connecting %s hand", side)
            backend.connect()

    def set_ready(self) -> None:
        for backend in self._backends.values():
            backend.set_position(list(self.cfg.ready_position))

    def set_positions(
        self,
        left: list[float] | None = None,
        right: list[float] | None = None,
    ) -> None:
        if left is not None and "left" in self._backends:
            self._backends["left"].set_position(left)
        if right is not None and "right" in self._backends:
            self._backends["right"].set_position(right)

    def read(self) -> dict:
        out: dict[str, dict] = {}
        for side, backend in self._backends.items():
            out[side] = {
                "position": backend.get_position(),
                "velocity": backend.get_velocity(),
                "current": backend.get_current(),
                "touch": backend.get_touch(),
            }
        return out

    def close(self) -> None:
        for side, backend in self._backends.items():
            logger.info("[PsyonicHands] closing %s hand", side)
            backend.close()


def make_psyonic_hands(cfg: PsyonicHandCfg) -> PsyonicHands | None:
    """Build a :class:`PsyonicHands`, or ``None`` if hands are disabled."""
    if not cfg.enabled:
        logger.info("[PsyonicHands] disabled (cfg.enabled=False); returning None")
        return None
    return PsyonicHands(cfg)


# ======================================================================
# Smoke test
# ======================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s"
    )

    cfg = PsyonicHandCfg(
        enabled=True,
        backend="mock",
        left_port="/dev/fake-left",
        right_port="/dev/fake-right",
    )

    hands = make_psyonic_hands(cfg)
    assert hands is not None
    print("present_sides:", hands.present_sides)

    hands.connect()
    hands.set_ready()
    print("commanded ready:", cfg.ready_position)

    hands.set_positions(
        left=[10.0, 20.0, 30.0, 40.0, 50.0, -20.0],
        right=[5.0, 5.0, 5.0, 5.0, 5.0, -5.0],
    )

    for i in range(4):
        state = hands.read()
        for side in hands.present_sides:
            pos = state[side]["position"]
            vel = state[side]["velocity"]
            print(
                f"read[{i}] {side}: pos="
                + "[" + ", ".join(f"{p:.2f}" for p in pos) + "]"
                + " vel="
                + "[" + ", ".join(f"{v:.2f}" for v in vel) + "]"
                + f" current={state[side]['current']} touch={state[side]['touch']}"
            )

    hands.close()

    # Disabled config returns None.
    assert make_psyonic_hands(PsyonicHandCfg(enabled=False)) is None
    print("smoke test OK")
