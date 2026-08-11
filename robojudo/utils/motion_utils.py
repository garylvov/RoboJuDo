# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Motion playback and heading-alignment utilities.

Vendored from the ProtoMotions ``deployment`` module so that RoboJuDo can
run inference without requiring the ProtoMotions source tree.

Quaternion convention: **xyzw** throughout (ProtoMotions common format).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

__all__ = [
    "MotionPlayer",
    "LiveRefSource",
    "compute_yaw_offset_np",
    "apply_heading_offset_np",
    "quat_mul_np",
    "quat_rotate_np",
    "_extract_yaw_quat_np",
]

# ---------------------------------------------------------------------------
# Quaternion helpers (pure NumPy)
# ---------------------------------------------------------------------------


def _extract_yaw_quat_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Extract the yaw-only quaternion from a full orientation (xyzw)."""
    x, y, z, w = q_xyzw[0], q_xyzw[1], q_xyzw[2], q_xyzw[3]
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = yaw * 0.5
    return np.array([0.0, 0.0, np.sin(half), np.cos(half)], dtype=np.float32)


def _quat_mul_np(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two xyzw quaternions (pure NumPy)."""
    ax, ay, az, aw = a_xyzw[..., 0], a_xyzw[..., 1], a_xyzw[..., 2], a_xyzw[..., 3]
    bx, by, bz, bw = b_xyzw[..., 0], b_xyzw[..., 1], b_xyzw[..., 2], b_xyzw[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1).astype(np.float32)


def _quat_conjugate_np(q_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for unit quats) of an xyzw quaternion."""
    result = q_xyzw.copy()
    result[..., :3] *= -1.0
    return result


def compute_yaw_offset_np(
    robot_quat_xyzw: np.ndarray,
    motion_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Compute a yaw-only heading offset between robot and motion frames.

    Returns a quaternion ``R_offset`` such that
    ``R_offset * motion_body_rot`` is in the robot's heading frame.
    """
    robot_yaw = _extract_yaw_quat_np(robot_quat_xyzw)
    motion_yaw = _extract_yaw_quat_np(motion_quat_xyzw)
    return _quat_mul_np(robot_yaw, _quat_conjugate_np(motion_yaw))


def apply_heading_offset_np(
    offset_quat_xyzw: np.ndarray,
    body_rots_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a heading offset to an array of body rotations.

    Computes ``offset * body_rot`` for every quaternion in the array.
    """
    original_shape = body_rots_xyzw.shape
    flat = body_rots_xyzw.reshape(-1, 4)
    offset_broadcast = np.broadcast_to(offset_quat_xyzw, flat.shape)
    aligned = _quat_mul_np(offset_broadcast, flat)
    return aligned.reshape(original_shape)


# Public alias -- exposed for policies that need to compose heading offsets
# with other quaternions (e.g. anchor-position displacement commands).
quat_mul_np = _quat_mul_np


def quat_rotate_np(q_xyzw: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector(s) ``v`` by unit quaternion(s) ``q_xyzw`` (Hamilton, xyzw).

    Broadcastable over leading batch dims; ``q_xyzw`` is broadcast to
    ``v``'s leading shape if it doesn't already match. Uses the standard
    cross-product formula (equivalent to the sandwich product
    ``q * (v, 0) * q_conj`` but without building 4-vectors).
    """
    q = np.broadcast_to(q_xyzw, v.shape[:-1] + (4,)).astype(np.float32)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    t = 2.0 * np.cross(q_vec, v)
    return (v + q_w * t + np.cross(q_vec, t)).astype(np.float32)


# ---------------------------------------------------------------------------
# MotionPlayer
# ---------------------------------------------------------------------------

_STATE_KEYS = ("dof_pos", "dof_vel", "body_rot", "body_pos", "body_vel", "body_ang_vel")


def _is_cache_file(data: dict) -> bool:
    """Return True if *data* looks like a pre-resampled cache."""
    return "control_dt" in data and "body_rot" in data


class MotionPlayer:
    """Lightweight player for a single motion clip at a fixed control rate.

    Accepts three input formats (auto-detected):

    1. Single ``.motion`` file -- RobotState dict with ``fps``, ``dof_pos``, etc.
    2. Packaged ``.pt`` library -- multi-motion with ``length_starts``, ``gts``, etc.
       Requires ``motion_index``.
    3. Pre-resampled cache -- written by :meth:`cache_to_file`.

    Formats 1 and 2 require ``protomotions`` for interpolation on first load.
    Format 3 (cached) is pure NumPy -- no external dependencies.
    """

    def __init__(
        self,
        motion_file: str,
        motion_index: int = 0,
        control_dt: float = 0.02,
    ):
        import torch

        self._torch = torch
        motion_file = str(motion_file)
        data = torch.load(motion_file, map_location="cpu", weights_only=False)

        if _is_cache_file(data):
            self._load_cache(data)
        else:
            self._load_raw(data, motion_index, control_dt)

    @property
    def total_frames(self) -> int:
        return self._num_frames

    @property
    def num_bodies(self) -> int:
        return self._body_rot.shape[1]

    @property
    def num_dofs(self) -> int:
        return self._dof_pos.shape[1]

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return the motion state at *frame_idx* (clamped)."""
        idx = int(np.clip(frame_idx, 0, self._num_frames - 1))
        return {
            "dof_pos":      self._dof_pos[idx],
            "dof_vel":      self._dof_vel[idx],
            "body_rot":     self._body_rot[idx],
            "body_pos":     self._body_pos[idx],
            "body_vel":     self._body_vel[idx],
            "body_ang_vel": self._body_ang_vel[idx],
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Return stacked future motion states at ``frame_idx + offset``."""
        future_states = [
            self.get_state_at_frame(frame_idx + s) for s in step_indices
        ]
        return {
            key: np.stack([s[key] for s in future_states], axis=0)
            for key in _STATE_KEYS
        }

    def cache_to_file(self, output_path: str) -> None:
        """Write a pre-resampled cache file at the current control rate."""
        import torch

        cache = {
            "dof_pos":      self._dof_pos,
            "dof_vel":      self._dof_vel,
            "body_rot":     self._body_rot,
            "body_pos":     self._body_pos,
            "body_vel":     self._body_vel,
            "body_ang_vel": self._body_ang_vel,
            "control_dt":   self._control_dt,
            "num_frames":   self._num_frames,
        }
        torch.save(cache, output_path)
        print(
            f"[MotionPlayer] Cached {self._num_frames} frames @ "
            f"{1.0 / self._control_dt:.0f} Hz -> {output_path}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_cache(self, data: dict) -> None:
        self._dof_pos      = np.asarray(data["dof_pos"],      dtype=np.float32)
        self._dof_vel      = np.asarray(data["dof_vel"],      dtype=np.float32)
        self._body_rot     = np.asarray(data["body_rot"],     dtype=np.float32)
        self._body_pos     = np.asarray(data["body_pos"],     dtype=np.float32)
        self._body_vel     = np.asarray(data["body_vel"],     dtype=np.float32)
        self._body_ang_vel = np.asarray(data["body_ang_vel"], dtype=np.float32)
        self._control_dt   = float(data["control_dt"])
        self._num_frames   = int(data["num_frames"])
        self._cached = True
        print(
            f"[MotionPlayer] Loaded cache: {self._num_frames} frames "
            f"@ {1.0 / self._control_dt:.0f} Hz"
        )

    def _load_raw(self, data: dict, motion_index: int, control_dt: float) -> None:
        """Load from a raw ProtoMotions motion file and resample."""
        self._control_dt = control_dt
        self._cached = False

        if "length_starts" in data:
            length_starts     = data["length_starts"]
            motion_num_frames = data["motion_num_frames"]
            motion_dt_all     = data["motion_dt"]

            start  = int(length_starts[motion_index].item())
            nf     = int(motion_num_frames[motion_index].item())
            src_dt = float(motion_dt_all[motion_index].item())

            gts  = np.asarray(data["gts"][start:start + nf],  dtype=np.float32)
            grs  = np.asarray(data["grs"][start:start + nf],  dtype=np.float32)
            gvs  = np.asarray(data["gvs"][start:start + nf],  dtype=np.float32)
            gavs = np.asarray(data["gavs"][start:start + nf], dtype=np.float32)
            dps  = np.asarray(data["dps"][start:start + nf],  dtype=np.float32)
            dvs  = np.asarray(data["dvs"][start:start + nf],  dtype=np.float32)
            motion_length = src_dt * (nf - 1)

        elif "rigid_body_pos" in data:
            fps    = float(data["fps"])
            src_dt = 1.0 / fps

            gts  = np.asarray(data["rigid_body_pos"],     dtype=np.float32)
            grs  = np.asarray(data["rigid_body_rot"],      dtype=np.float32)
            gvs  = np.asarray(data["rigid_body_vel"],      dtype=np.float32)
            gavs = np.asarray(data["rigid_body_ang_vel"],  dtype=np.float32)
            dps  = np.asarray(data["dof_pos"],             dtype=np.float32)
            dvs  = np.asarray(data["dof_vel"],             dtype=np.float32)
            nf   = gts.shape[0]
            motion_length = src_dt * (nf - 1)
        else:
            raise ValueError(
                "Unrecognised raw motion format.  Expected either:\n"
                "  - packaged library: keys 'length_starts', 'gts', 'grs', ...\n"
                "  - single-motion:   keys 'rigid_body_pos', 'fps', 'dof_pos', ..."
            )

        # Resample to control rate (lerp for positions, slerp for quaternions)
        num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)
        ctrl_times = np.linspace(0.0, motion_length, num_ctrl_frames)

        phase = np.clip(ctrl_times / motion_length, 0.0, 1.0)
        f0 = (phase * (nf - 1)).astype(np.int64)
        f1 = np.minimum(f0 + 1, nf - 1)
        blend = ((ctrl_times - f0 * src_dt) / src_dt).astype(np.float32)

        def _lerp(src):
            b = blend.reshape(-1, *([1] * (src.ndim - 1)))
            return ((1.0 - b) * src[f0] + b * src[f1]).astype(np.float32)

        def _slerp(src):
            q0, q1 = src[f0], src[f1]
            cos_half = np.sum(q0 * q1, axis=-1, keepdims=True)
            # Flip to shortest path
            neg = cos_half < 0
            q1 = np.where(neg, -q1, q1)
            cos_half = np.abs(cos_half)
            half_theta = np.arccos(np.clip(cos_half, -1.0, 1.0))
            sin_half = np.sqrt(np.maximum(1.0 - cos_half * cos_half, 0.0))
            b = blend.reshape(-1, *([1] * (q0.ndim - 1)))
            # Safe divide — degenerate cases handled by np.where below
            safe_sin = np.where(sin_half > 0, sin_half, 1.0)
            ratio_a = np.sin((1.0 - b) * half_theta) / safe_sin
            ratio_b = np.sin(b * half_theta) / safe_sin
            result = ratio_a * q0 + ratio_b * q1
            # Fallback: near-zero sin_half → linear blend; cos_half ≈ 1 → q0
            near_zero = np.abs(sin_half) < 0.001
            linear = 0.5 * q0 + 0.5 * q1
            result = np.where(near_zero, linear, result)
            identical = np.abs(cos_half) >= 1.0
            result = np.where(identical, q0, result)
            return result.astype(np.float32)

        self._body_pos     = _lerp(gts)
        self._body_rot     = _slerp(grs)
        self._body_vel     = _lerp(gvs)
        self._body_ang_vel = _lerp(gavs)
        self._dof_pos      = _lerp(dps)
        self._dof_vel      = _lerp(dvs)
        self._num_frames   = num_ctrl_frames

        print(
            f"[MotionPlayer] Loaded raw motion #{motion_index}: "
            f"{nf} source frames @ {1.0 / src_dt:.1f} Hz -> "
            f"{num_ctrl_frames} resampled frames @ {1.0 / control_dt:.0f} Hz"
        )


# ---------------------------------------------------------------------------
# LiveRefSource  (imprint teleop integration -- additive)
# ---------------------------------------------------------------------------


class LiveRefSource:
    """Live (streaming) reference source that duck-types :class:`MotionPlayer`.

    Instead of playing back a pre-recorded clip, this buffers the LATEST
    reference produced by an external teleop retargeter (see
    ``imprint.robojudo.teleop``) and serves it to the tracker policy in place
    of a ``MotionPlayer``.  It exposes the subset of the ``MotionPlayer`` API
    that ``ProtoMotionsTrackerPolicy`` reads:

    - ``total_frames`` -- a very large constant (the stream never ends)
    - ``get_state_at_frame(frame)`` -- returns the latest buffered reference
      (the ``frame`` argument is ignored; there is no history)
    - ``get_future_references(frame, step_indices)`` -- the latest reference
      replicated ``len(step_indices)`` times (a zero-order hold: we have no
      look-ahead for a live stream)
    - ``update(dof_pos, dof_vel, body_rot)`` -- push a new reference

    Before the first :meth:`update`, a seeded default is returned:
    ``dof_pos = default_dof_pos`` (or zeros), ``dof_vel = 0`` and per-body
    identity rotations (xyzw ``[0, 0, 0, 1]``), so the policy can run in the
    hold-default-pose UX before any teleop command arrives.

    Quaternion convention: **xyzw** (matches ``MotionPlayer``).
    """

    _HUGE_FRAMES = 1 << 30

    def __init__(
        self,
        num_dofs: int = 27,
        num_bodies: int = 29,
        anchor_idx: int = 0,
        default_dof_pos: np.ndarray | None = None,
        control_dt: float = 0.02,
    ):
        self._num_dofs = int(num_dofs)
        self._num_bodies = int(num_bodies)
        self._anchor_idx = int(anchor_idx)
        self._control_dt = float(control_dt)

        if default_dof_pos is None:
            seed_dof_pos = np.zeros(self._num_dofs, dtype=np.float32)
        else:
            seed_dof_pos = np.asarray(default_dof_pos, dtype=np.float32).reshape(-1)

        # Seeded default reference (identity per-body rotation, zero velocity).
        identity_rot = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (self._num_bodies, 1),
        )
        self._dof_pos = seed_dof_pos.copy()
        self._dof_vel = np.zeros(self._num_dofs, dtype=np.float32)
        self._body_rot = identity_rot
        # Optional operator-as-odom anchor position channel (imprint teleop).
        # None until the first update() that carries anchor_pos; then the
        # served states grow a "body_pos" array whose ANCHOR row is this
        # position (the only row the tracker reads -- see
        # ProtoMotionsTrackerPolicy's displacement command). Future references
        # extrapolate with an EMA'd finite-difference velocity, since a live
        # stream has no real look-ahead and a zero-order hold would zero the
        # (future - current) displacement command. ROOT channel only: joint
        # references are untouched.
        self._anchor_pos: np.ndarray | None = None
        self._anchor_vel = np.zeros(3, dtype=np.float32)
        self._anchor_vel_ema_alpha = 0.2
        # Operator-as-odom look-ahead buffer. The tracker's actor conditions
        # on FUTURE references (future_dof_pos/dof_vel/anchor_rot at +1..+8
        # steps); with a pure zero-order hold the "future" equals the present,
        # so the policy sees a reference that is static over its horizon and
        # station-keeps -- the walking gait in the dofs never becomes a
        # locomotion command. When odom is active (anchor_pos fed), the served
        # "current" reference is DELAYED by up to `_lookahead_steps` ticks and
        # the newer samples become genuine futures -- the same leading
        # reference playback feeds, at the cost of ~0.16 s of teleop latency.
        # Flushed on engage-generation change so a re-clutch never serves a
        # pre-engage reference against a post-engage alignment.
        self._lookahead_steps = 8
        self._history: list[dict] = []
        self._history_generation = None

    @property
    def total_frames(self) -> int:
        return self._HUGE_FRAMES

    @property
    def num_bodies(self) -> int:
        return self._num_bodies

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    def update(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        body_rot: np.ndarray,
        anchor_pos: np.ndarray | None = None,
        engage_generation=None,
    ) -> None:
        """Push the latest retargeted reference.

        ``anchor_pos`` (3,) is the optional operator-as-odom anchor position
        (a stitched-CONTINUOUS trajectory by the provider's contract -- it
        never jumps across a clutch, so the finite-difference velocity below
        never spikes). ``None`` keeps the previous behaviour exactly: latest
        sample only, zero-order hold, no position channel served.

        With ``anchor_pos``, samples also enter the look-ahead buffer (see
        ``__init__``); ``engage_generation`` flushes it on change so the
        served "current" can never be a pre-engage sample.
        """
        self._dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(self._num_dofs)
        self._dof_vel = np.asarray(dof_vel, dtype=np.float32).reshape(self._num_dofs)
        self._body_rot = np.asarray(body_rot, dtype=np.float32).reshape(
            self._num_bodies, 4
        )
        if anchor_pos is None:
            return
        p = np.asarray(anchor_pos, dtype=np.float32).reshape(3)
        if self._anchor_pos is None:
            self._anchor_vel = np.zeros(3, dtype=np.float32)
        else:
            raw_vel = (p - self._anchor_pos) / self._control_dt
            a = self._anchor_vel_ema_alpha
            self._anchor_vel = (
                a * raw_vel + (1.0 - a) * self._anchor_vel
            ).astype(np.float32)
        self._anchor_pos = p

        if engage_generation != self._history_generation:
            self._history_generation = engage_generation
            self._history = []
            self._anchor_vel = np.zeros(3, dtype=np.float32)
        self._history.append(
            {
                "dof_pos": self._dof_pos,
                "dof_vel": self._dof_vel,
                "body_rot": self._body_rot,
                "anchor_pos": p,
            }
        )
        if len(self._history) > self._lookahead_steps + 1:
            self._history.pop(0)

    def _anchor_body_pos(self, p: np.ndarray) -> np.ndarray:
        """(num_bodies, 3) with every row = ``p``.

        Only the ANCHOR row is meaningful (the tracker reads nothing else);
        the other rows repeat it so the shape duck-types MotionPlayer's.
        """
        return np.broadcast_to(p, (self._num_bodies, 3)).astype(np.float32)

    def _sample(self, offset: int) -> dict:
        """History sample ``offset`` steps ahead of the served "current"."""
        idx = min(offset, len(self._history) - 1)
        return self._history[idx]

    def get_state_at_frame(self, frame_idx: int) -> Dict[str, np.ndarray]:
        """Return the served "current" reference (``frame_idx`` is ignored).

        Without odom this is the LATEST buffered sample (previous behaviour).
        With odom it is the OLDEST sample in the look-ahead buffer -- the
        reference runs ``len(history)-1`` ticks (up to 0.16 s) behind the
        operator so the newer samples can serve as genuine futures.
        """
        if not self._history:
            return {
                "dof_pos":  self._dof_pos,
                "dof_vel":  self._dof_vel,
                "body_rot": self._body_rot,
            }
        cur = self._sample(0)
        return {
            "dof_pos":  cur["dof_pos"],
            "dof_vel":  cur["dof_vel"],
            "body_rot": cur["body_rot"],
            "body_pos": self._anchor_body_pos(cur["anchor_pos"]),
        }

    def get_future_references(
        self,
        frame_idx: int,
        step_indices: List[int],
    ) -> Dict[str, np.ndarray]:
        """Future references ``step_indices`` ticks ahead of the current.

        Without odom: the latest reference replicated (previous behaviour).
        With odom: genuine future samples from the look-ahead buffer; steps
        beyond the newest sample hold it, with the anchor POSITION alone
        extrapolated by the EMA'd anchor velocity (ROOT channel only -- joint
        references are never extrapolated).
        """
        n = len(step_indices)
        if not self._history:
            return {
                "dof_pos":  np.broadcast_to(self._dof_pos, (n, self._num_dofs)).copy(),
                "dof_vel":  np.broadcast_to(self._dof_vel, (n, self._num_dofs)).copy(),
                "body_rot": np.broadcast_to(
                    self._body_rot, (n, self._num_bodies, 4)
                ).copy(),
            }
        newest = len(self._history) - 1
        samples = [self._sample(int(step)) for step in step_indices]
        body_pos = []
        for step, s in zip(step_indices, samples):
            p = s["anchor_pos"]
            overshoot = int(step) - newest
            if overshoot > 0:
                p = p + self._anchor_vel * (overshoot * self._control_dt)
            body_pos.append(self._anchor_body_pos(p))
        return {
            "dof_pos":  np.stack([s["dof_pos"] for s in samples]),
            "dof_vel":  np.stack([s["dof_vel"] for s in samples]),
            "body_rot": np.stack([s["body_rot"] for s in samples]),
            "body_pos": np.stack(body_pos),
        }

