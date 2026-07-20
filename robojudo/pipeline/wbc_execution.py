"""Whole-body-controller (WBC) execution state machine.

A single coherent state machine that governs *whether* and *how* policy
actions reach the robot.  It composes the deploy-critical control primitives —
freeze / resume / damping / stepped execution / action-preview — instead of
bolting them on as independent flags.

States (:class:`ExecMode`)::

    RUNNING     continuous policy stepping (normal operation)
    FROZEN      hold latched pose; policy frame paused; normal gains
    PREVIEW     hold latched pose; a pending action has been computed from the
                current observation and surfaced (logged/serialized) but NOT
                executed; awaiting CONFIRM
    STEPPING    finite budget of steps to execute, then auto-freeze
                (single-step = budget 1, burst = budget N)
    RAMP        smoothly interpolate the held target from a start pose to a
                goal pose (used for startup ready-pose and HANDS_READY), then
                auto-freeze at the goal
    DAMPING     hold; damping mode engaged on hardware (soft); policy paused

Transitions (driven by pipeline commands)::

    [FREEZE_WBC]            *            -> FROZEN   (latch current target)
    [RESUME_POLICY]         non-RUNNING  -> RUNNING  (resync obs first)
    [POLICY_RUN_CONTINUOUS] *            -> RUNNING  (alias of resume)
    [DAMPING]               *            -> DAMPING  (set_damping_mode on real)
    [HANDS_READY]           *            -> RAMP(ready_pose) -> FROZEN
    [POLICY_PREVIEW]        RUNNING|FROZEN -> PREVIEW (compute+surface pending)
    [POLICY_CONFIRM]        PREVIEW      -> execute pending once -> FROZEN
    [POLICY_STEP_ONCE]      *            -> STEPPING(1) -> FROZEN
    [POLICY_STEP_BURST]     *            -> STEPPING(N) -> FROZEN

The stepped modes auto-enter FROZEN between/after steps, so single-step, burst
and preview all leave the robot in a safe held state by default.  Policy
engagement is always a deliberate act (RESUME / CONTINUOUS / STEP / CONFIRM).
"""

from __future__ import annotations

import enum
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from robojudo.config import Config

logger = logging.getLogger(__name__)


# ==== Command tokens owned by this state machine ====
CMD_FREEZE = "[FREEZE_WBC]"
CMD_RESUME = "[RESUME_POLICY]"
CMD_DAMPING = "[DAMPING]"
CMD_HANDS_READY = "[HANDS_READY]"
CMD_PREVIEW = "[POLICY_PREVIEW]"
CMD_CONFIRM = "[POLICY_CONFIRM]"
CMD_STEP_ONCE = "[POLICY_STEP_ONCE]"
CMD_STEP_BURST = "[POLICY_STEP_BURST]"
CMD_RUN_CONTINUOUS = "[POLICY_RUN_CONTINUOUS]"

WBC_COMMANDS = frozenset(
    {
        CMD_FREEZE,
        CMD_RESUME,
        CMD_DAMPING,
        CMD_HANDS_READY,
        CMD_PREVIEW,
        CMD_CONFIRM,
        CMD_STEP_ONCE,
        CMD_STEP_BURST,
        CMD_RUN_CONTINUOUS,
    }
)


class ExecMode(enum.Enum):
    RUNNING = "RUNNING"
    FROZEN = "FROZEN"
    PREVIEW = "PREVIEW"
    STEPPING = "STEPPING"
    RAMP = "RAMP"
    DAMPING = "DAMPING"


class WbcExecCfg(Config):
    """Config for the WBC execution state machine."""

    enabled: bool = True
    """If False the controller is a transparent pass-through in RUNNING mode
    (it still accepts commands, but no startup gating is applied)."""

    startup_ready_pose: bool = False
    """If True the pipeline ramps to the ready pose and holds FROZEN at startup
    (policy NOT running).  Deploy-default; off for legacy sim demos."""

    burst_steps: int = 10
    """Number of policy steps executed by a [POLICY_STEP_BURST] command."""

    ramp_seconds: float = 2.0
    """Duration of the smooth interpolation for HANDS_READY / startup ramps."""

    resync_on_resume: bool = True
    """Re-sync the policy observation/alignment before unfreezing on resume."""

    preview_dir: str = "logs/preview"
    """Directory where pending-action previews are serialized for offline viz."""

    serialize_preview: bool = True
    """If True, PREVIEW writes a JSON file with the pending action + deltas."""


@dataclass
class ExecDecision:
    """What the pipeline should do with this step."""

    pd_target: np.ndarray
    advance_policy: bool  # whether the policy frame advances (executed step)
    executed: bool  # whether a real policy action reached the robot this step
    mode: ExecMode


@dataclass
class _RampState:
    start: np.ndarray
    goal: np.ndarray
    total_steps: int
    step: int = 0
    tag: str = "ramp"
    then_running: bool = False  # RAMP -> RUNNING (else -> FROZEN)


class WbcExecutionController:
    """Owns the WBC execution mode and resolves each pipeline step.

    Parameters
    ----------
    cfg:
        :class:`WbcExecCfg`.
    freq:
        Policy frequency (Hz), used to convert ramp seconds -> steps.
    env:
        Environment; used for ``set_damping_mode`` / ``set_gains`` if present.
    resync_cb:
        Optional callable invoked before unfreezing on resume (e.g. the
        pipeline's ``policy.reset_alignment``).  May be ``None``.
    """

    def __init__(self, cfg: WbcExecCfg, freq: float, env=None, resync_cb=None):
        self.cfg = cfg
        self.freq = float(freq)
        self.env = env
        self.resync_cb = resync_cb

        self.mode = ExecMode.RUNNING
        self._latched: np.ndarray | None = None
        self._step_budget = 0
        self._pending: np.ndarray | None = None
        self._pending_info: dict | None = None
        self._ramp: _RampState | None = None
        self._preview_seq = 0

        self._preview_dir = Path(cfg.preview_dir)

    # ------------------------------------------------------------------ #
    # Command handling
    # ------------------------------------------------------------------ #
    def handle_commands(self, commands, current_dof: np.ndarray, ready_pose: np.ndarray | None = None) -> list[str]:
        """Consume WBC commands, return the list of commands actually handled."""
        handled = []
        for cmd in commands:
            if cmd not in WBC_COMMANDS:
                continue
            handled.append(cmd)
            self._apply_command(cmd, current_dof, ready_pose)
        return handled

    def _apply_command(self, cmd: str, current_dof: np.ndarray, ready_pose: np.ndarray | None):
        cur = np.asarray(current_dof, dtype=np.float32)
        if cmd == CMD_FREEZE:
            self._enter_frozen(self._latched if self._latched is not None else cur, reason="FREEZE_WBC")
        elif cmd in (CMD_RESUME, CMD_RUN_CONTINUOUS):
            self._enter_running()
        elif cmd == CMD_DAMPING:
            self._enter_damping(cur)
        elif cmd == CMD_HANDS_READY:
            goal = ready_pose if ready_pose is not None else cur
            self._start_ramp(cur, np.asarray(goal, dtype=np.float32), tag="hands_ready", then_running=False)
        elif cmd == CMD_PREVIEW:
            # Actual pending action is captured in tick() (needs policy_pd).
            self._latched = self._latched if self._latched is not None else cur
            self.mode = ExecMode.PREVIEW
            self._pending = None  # recomputed on next tick
            logger.info("[WBC] PREVIEW requested — computing pending action, robot held frozen")
        elif cmd == CMD_CONFIRM:
            if self.mode != ExecMode.PREVIEW or self._pending is None:
                logger.warning("[WBC] CONFIRM ignored — no pending previewed action")
            else:
                logger.info("[WBC] CONFIRM — executing previewed action once, then FREEZE")
                # tick() will send self._pending with advance=True then FROZEN
                self._confirm_pending = True
        elif cmd == CMD_STEP_ONCE:
            self._step_budget = 1
            self.mode = ExecMode.STEPPING
            logger.info("[WBC] SINGLE-STEP — execute 1 policy step then auto-freeze")
        elif cmd == CMD_STEP_BURST:
            self._step_budget = max(1, int(self.cfg.burst_steps))
            self.mode = ExecMode.STEPPING
            logger.info(f"[WBC] BURST — execute {self._step_budget} policy steps then auto-freeze")

    # ---- transition helpers ----
    _confirm_pending = False

    def _enter_running(self):
        if self.mode != ExecMode.RUNNING and self.cfg.resync_on_resume and self.resync_cb is not None:
            try:
                self.resync_cb()
                logger.info("[WBC] resync policy observation/alignment before resume")
            except Exception as e:  # pragma: no cover - defensive
                logger.warning(f"[WBC] resync callback failed: {e}")
        if self.mode == ExecMode.DAMPING:
            self._restore_gains()
        self.mode = ExecMode.RUNNING
        self._step_budget = 0
        self._pending = None
        self._ramp = None
        self._confirm_pending = False
        logger.info("[WBC] -> RUNNING (continuous policy stepping)")

    def _enter_frozen(self, target: np.ndarray, reason: str = "freeze"):
        if self.mode == ExecMode.DAMPING:
            self._restore_gains()
        self._latched = np.asarray(target, dtype=np.float32).copy()
        self.mode = ExecMode.FROZEN
        self._step_budget = 0
        self._ramp = None
        self._confirm_pending = False
        logger.info(f"[WBC] -> FROZEN ({reason}); holding latched pose")

    def _enter_damping(self, current_dof: np.ndarray):
        self._latched = np.asarray(current_dof, dtype=np.float32).copy()
        self.mode = ExecMode.DAMPING
        self._step_budget = 0
        self._ramp = None
        env = self.env
        if env is not None and hasattr(env, "set_damping_mode"):
            try:
                env.set_damping_mode()
                logger.warning("[WBC] -> DAMPING; hardware set_damping_mode() engaged (soft)")
            except Exception as e:  # pragma: no cover - hardware path
                logger.error(f"[WBC] set_damping_mode failed: {e}")
        else:
            logger.warning("[WBC] -> DAMPING; no hardware damping on this env (mock/sim) — no-op, holding pose")

    def _restore_gains(self):
        env = self.env
        if env is not None and hasattr(env, "set_gains"):
            try:
                env.set_gains(env.stiffness, env.damping)
                logger.info("[WBC] restored normal gains after damping")
            except Exception as e:  # pragma: no cover
                logger.warning(f"[WBC] set_gains restore failed: {e}")

    def _start_ramp(self, start: np.ndarray, goal: np.ndarray, tag: str, then_running: bool):
        steps = max(1, int(self.cfg.ramp_seconds * self.freq))
        self._ramp = _RampState(
            start=np.asarray(start, dtype=np.float32).copy(),
            goal=np.asarray(goal, dtype=np.float32).copy(),
            total_steps=steps,
            tag=tag,
            then_running=then_running,
        )
        self.mode = ExecMode.RAMP
        logger.info(f"[WBC] -> RAMP ({tag}); interpolating over {steps} steps ({self.cfg.ramp_seconds:.1f}s)")

    def start_ready_ramp(self, current_dof: np.ndarray, ready_pose: np.ndarray, then_running: bool = False):
        """Public entry used by pipeline startup() to ramp to the ready pose."""
        self._start_ramp(current_dof, ready_pose, tag="startup_ready", then_running=then_running)

    def force_freeze(self, target: np.ndarray, reason: str = "external"):
        """Public: latch ``target`` and enter FROZEN (used by pipeline startup)."""
        self._enter_frozen(np.asarray(target, dtype=np.float32), reason=reason)

    # ------------------------------------------------------------------ #
    # Per-step resolution
    # ------------------------------------------------------------------ #
    def tick(self, policy_pd: np.ndarray, current_dof: np.ndarray, timestep: int = 0) -> ExecDecision:
        """Resolve the final pd_target + whether the policy frame advances."""
        policy_pd = np.asarray(policy_pd, dtype=np.float32)

        if self.mode == ExecMode.RUNNING:
            self._latched = policy_pd.copy()
            return ExecDecision(policy_pd, advance_policy=True, executed=True, mode=self.mode)

        if self.mode == ExecMode.FROZEN:
            return ExecDecision(self._held(current_dof), advance_policy=False, executed=False, mode=self.mode)

        if self.mode == ExecMode.DAMPING:
            return ExecDecision(self._held(current_dof), advance_policy=False, executed=False, mode=self.mode)

        if self.mode == ExecMode.PREVIEW:
            # Capture / refresh the pending previewed action from current obs.
            if self._confirm_pending and self._pending is not None:
                pending = self._pending.copy()
                self._enter_frozen(pending, reason="confirmed action")
                return ExecDecision(pending, advance_policy=True, executed=True, mode=ExecMode.FROZEN)
            if self._pending is None:
                self._pending = policy_pd.copy()
                self._surface_preview(policy_pd, current_dof, timestep)
            return ExecDecision(self._held(current_dof), advance_policy=False, executed=False, mode=self.mode)

        if self.mode == ExecMode.STEPPING:
            if self._step_budget > 0:
                self._latched = policy_pd.copy()
                self._step_budget -= 1
                if self._step_budget == 0:
                    logger.info("[WBC] step budget exhausted -> auto-FREEZE after this step")
                    # execute this step, then next tick will be FROZEN
                    self.mode = ExecMode.FROZEN
                    self._latched = policy_pd.copy()
                return ExecDecision(policy_pd, advance_policy=True, executed=True, mode=ExecMode.STEPPING)
            # nothing left (defensive): freeze
            self._enter_frozen(self._held(current_dof), reason="stepping-empty")
            return ExecDecision(self._held(current_dof), advance_policy=False, executed=False, mode=self.mode)

        if self.mode == ExecMode.RAMP:
            r = self._ramp
            assert r is not None
            alpha = min((r.step + 1) / max(r.total_steps, 1), 1.0)
            target = (1.0 - alpha) * r.start + alpha * r.goal
            r.step += 1
            if r.step >= r.total_steps:
                if r.then_running:
                    logger.info(f"[WBC] RAMP ({r.tag}) done -> RUNNING")
                    self.mode = ExecMode.RUNNING
                    self._latched = r.goal.copy()
                else:
                    self._enter_frozen(r.goal, reason=f"{r.tag} reached")
                self._ramp = None
            return ExecDecision(target.astype(np.float32), advance_policy=False, executed=False, mode=ExecMode.RAMP)

        # Fallback
        return ExecDecision(self._held(current_dof), advance_policy=False, executed=False, mode=self.mode)

    def _held(self, current_dof: np.ndarray) -> np.ndarray:
        if self._latched is not None:
            return self._latched.copy()
        return np.asarray(current_dof, dtype=np.float32).copy()

    # ------------------------------------------------------------------ #
    # Preview serialization
    # ------------------------------------------------------------------ #
    def _surface_preview(self, pending_pd: np.ndarray, current_dof: np.ndarray, timestep: int):
        cur = np.asarray(current_dof, dtype=np.float32)
        delta = np.asarray(pending_pd, dtype=np.float32) - cur
        info = {
            "timestep": int(timestep),
            "time": time.time(),
            "seq": self._preview_seq,
            "current_dof_pos": cur.tolist(),
            "pending_pd_target": np.asarray(pending_pd, dtype=np.float32).tolist(),
            "delta": delta.tolist(),
            "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
            "argmax_delta_joint": int(np.argmax(np.abs(delta))) if delta.size else -1,
        }
        self._pending_info = info
        logger.warning(
            f"[WBC] PREVIEW pending action seq={self._preview_seq} "
            f"max|Δ|={info['max_abs_delta']:.4f} rad @ joint {info['argmax_delta_joint']} "
            "(NOT executed — send CONFIRM to apply)"
        )
        if self.cfg.serialize_preview:
            try:
                os.makedirs(self._preview_dir, exist_ok=True)
                path = self._preview_dir / f"preview_{time.strftime('%y%m%d-%H%M%S')}_{self._preview_seq:04d}.json"
                with open(path, "w") as f:
                    json.dump(info, f, indent=2)
                logger.info(f"[WBC] preview serialized -> {path}")
            except Exception as e:  # pragma: no cover
                logger.warning(f"[WBC] failed to serialize preview: {e}")
        self._preview_seq += 1

    def get_pending_preview(self) -> dict | None:
        """Return the most recent pending previewed action info (or None)."""
        return self._pending_info

    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        return {
            "mode": self.mode.value,
            "step_budget": self._step_budget,
            "has_pending": self._pending is not None,
            "ramp_active": self._ramp is not None,
        }
