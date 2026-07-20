import logging
import time

import numpy as np
from box import Box

import robojudo.environment
import robojudo.policy
from robojudo.controller import CtrlManager
from robojudo.environment import Environment
from robojudo.environment.psyonic_hand import make_psyonic_hands
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.wbc_execution import ExecMode, WbcExecCfg, WbcExecutionController
from robojudo.policy import Policy, PolicyCfg
from robojudo.tools.dof import DoFAdapter
from robojudo.tools.recorder import RateRecorder, RecorderCfg
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.progress import ProgressBar
from robojudo.utils.util_func import get_gravity_orientation

logger = logging.getLogger(__name__)


class PolicyWrapper:
    """A wrapper for Policy to handle observation and action adaptation."""

    def __init__(self, cfg_policy: PolicyCfg, env_dof_cfg: DoFConfig, device: str):
        self.env_dof_cfg = env_dof_cfg

        policy_type = cfg_policy.policy_type
        policy_name = policy_type
        if hasattr(cfg_policy, "policy_name"):
            policy_name += "@" + cfg_policy.policy_name  # type: ignore
        # while policy_name in self.policies.keys():
        #     policy_name += "_new"
        self.name = policy_name

        policy_class: type[Policy] = getattr(robojudo.policy, policy_type)
        self.policy: Policy = policy_class(cfg_policy=cfg_policy, device=device)
        self.obs_adapter = DoFAdapter(env_dof_cfg.joint_names, self.policy.cfg_obs_dof.joint_names)
        self.actions_adapter = DoFAdapter(self.policy.cfg_action_dof.joint_names, env_dof_cfg.joint_names)

    def get_observation(self, env_data: Box, ctrl_data: Box):
        env_data_adapted = env_data.copy()
        env_data_adapted.dof_pos = self.obs_adapter.fit(env_data_adapted.dof_pos)
        env_data_adapted.dof_vel = self.obs_adapter.fit(env_data_adapted.dof_vel)
        return self.policy.get_observation(env_data_adapted, ctrl_data)

    def get_action(self, obs):
        action = self.policy.get_action(obs)
        return self.actions_adapter.fit(action)

    def get_pd_target(self, obs):
        action = self.policy.get_action(obs)
        pd_target = action + self.policy.default_pos
        return self.actions_adapter.fit(pd_target, template=self.env_dof_cfg.default_pos)

    def get_init_dof_pos(self):
        return self.actions_adapter.fit(self.policy.get_init_dof_pos(), template=self.env_dof_cfg.default_pos)

    def __getattr__(self, name):
        """Fallback: delegate other func to the wrapped policy."""
        return getattr(self.policy, name)


@pipeline_registry.register
class RlPipeline(Pipeline):
    cfg: RlPipelineCfg

    def __init__(self, cfg: RlPipelineCfg):
        super().__init__(cfg=cfg)

        env_class: type[Environment] = getattr(robojudo.environment, self.cfg.env.env_type)
        self.env: Environment = env_class(cfg_env=self.cfg.env, device=self.device)

        self.ctrl_manager = CtrlManager(cfg_ctrls=self.cfg.ctrl, env=self.env, device=self.device)

        self.policy = PolicyWrapper(
            cfg_policy=self.cfg.policy,
            env_dof_cfg=self.env.dof_cfg,
            device=self.device,
        )

        self.env.update_dof_cfg(override_cfg=self.policy.cfg_action_dof)
        self.visualizer = self.env.visualizer

        self.freq = self.cfg.policy.freq
        self.dt = 1.0 / self.freq

        # ===== Deploy control-flow: WBC state machine, recorder, hands =====
        wbc_cfg: WbcExecCfg = getattr(self.cfg, "wbc", None) or WbcExecCfg()
        self.exec = WbcExecutionController(
            cfg=wbc_cfg,
            freq=self.freq,
            env=self.env,
            resync_cb=self._resync_policy,
        )
        rec_cfg: RecorderCfg = getattr(self.cfg, "recorder", None) or RecorderCfg()
        self.recorder = RateRecorder(rec_cfg, run_name=type(self.cfg).__name__)
        self.hands = make_psyonic_hands(getattr(self.cfg, "psyonic", None) or None)
        if self.hands is not None:
            try:
                self.hands.connect()
                logger.info(f"[Pipeline] Psyonic hands connected: {self.hands.present_sides}")
            except Exception as e:  # pragma: no cover - hardware path
                logger.error(f"[Pipeline] Psyonic hand connect failed: {e}")
                self.hands = None

        # Ready pose (full DoF) for startup ramp + HANDS_READY.
        ready = getattr(self.env.cfg_env, "ready_pose", None)
        if ready is not None and len(ready) == self.env.num_dofs:
            self._ready_pose = np.asarray(ready, dtype=np.float32)
        else:
            if ready is not None:
                logger.warning(
                    f"[Pipeline] env.ready_pose len {len(ready)} != num_dofs "
                    f"{self.env.num_dofs}; falling back to default_pos"
                )
            self._ready_pose = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32)

        self.reset()
        self.self_check()
        self.policy.reset()  # reset frame counter after dry-run steps

    def self_check(self):
        self.env.self_check()
        for _ in range(10):
            self.step(dry_run=True)

    def _inner_policy(self):
        """Return the unwrapped inner Policy (e.g. ProtoMotionsTrackerPolicy)."""
        return getattr(self.policy, "policy", self.policy)

    def _resync_policy(self):
        """Safe transition hook: re-sync policy observation/alignment before
        unfreezing (called by the WBC controller on [RESUME_POLICY]).

        This recomputes any runtime heading/spatial alignment so the policy
        resumes from the robot's CURRENT pose rather than a stale reference,
        avoiding a jerk when leaving a frozen/stepped state.
        """
        inner = self._inner_policy()
        if hasattr(inner, "reset_alignment"):
            inner.reset_alignment()
        # Clear any lingering pause so the frame advances again.
        if hasattr(inner, "_paused"):
            inner._paused = False

    @property
    def _has_default_pose_mode(self) -> bool:
        return hasattr(self._inner_policy(), "set_default_pose_mode")

    def _set_default_pose_mode(self, enabled: bool):
        """Enable/disable default-pose mode on the inner policy (if supported)."""
        inner = self._inner_policy()
        if hasattr(inner, "set_default_pose_mode"):
            inner.set_default_pose_mode(enabled)

    def reset(self):
        logger.info("Pipeline reset")
        self.timestep = 0

        self.env.reset()
        self.policy.reset()
        self.ctrl_manager.reset()

        # Blend-out state: transitions policy → init pose at end of motion.
        self._blend_out_active = False
        self._blend_out_step = 0
        self._blend_out_duration = int(5.0 * self.freq)  # 5 seconds

        # For tracker policies with default-pose mode, ramp/blend target is
        # the env's default standing pose.  Otherwise, use motion frame 0.
        if self._has_default_pose_mode:
            self._init_dof_pos = np.asarray(self.env.dof_cfg.default_pos, dtype=np.float32)
        else:
            self._init_dof_pos = np.asarray(self.policy.get_init_dof_pos(), dtype=np.float32)

        self._pending_blend_in = False
        self._blend_in_completed = False
        self._user_fade_out = False  # True when fade-out was user-triggered (not auto)
        self._prepare_seconds = None  # set by prepare() for re-use on reset

    def safety_check(self):
        if not self.do_safety_check:
            return
        gravity_ori = get_gravity_orientation(self.env.base_quat)
        angle = np.arccos(np.clip(-gravity_ori[2], -1.0, 1.0))
        if abs(angle) > 1.0:  # more than ~57 degrees
            logger.error("Robot fallen! Shutdown for safety.")
            if hasattr(self.env, "reborn"):
                self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                self.policy.reset_alignment()
            else:
                self.env.shutdown()

    def post_step_callback(self, env_data, ctrl_data, extras, pd_target):
        self.timestep += 1
        commands = ctrl_data.get("COMMANDS", [])
        for command in commands:
            match command:
                case "[SHUTDOWN]":
                    logger.warning("Emergency shutdown!")
                    self.env.shutdown()
                    self._teardown_deploy_resources()
                case "[SIM_REBORN]":
                    if hasattr(self.env, "reborn"):
                        logger.warning("Simulation Env reborn!")
                        self.env.reborn()  # pyright: ignore[reportAttributeAccessIssue]
                        self.policy.reset_alignment()
                case "[MOTION_RESET]" | "[MOTION_FADE_IN]":
                    self._blend_out_active = False
                    self._blend_out_step = 0
                    self._user_fade_out = False
                    if self._has_default_pose_mode:
                        # Policy is already active — just switch target
                        # from default pose to motion (instant, no blend).
                        logger.info(
                            f"{command} — starting motion from frame 0"
                        )
                        self._set_default_pose_mode(False)
                    else:
                        # Legacy path: full blend-in needed.
                        logger.info(
                            f"{command} — re-entering blend-in phase"
                        )
                        self._pending_blend_in = True
                case "[MOTION_FADE_OUT]":
                    if self._has_default_pose_mode:
                        logger.info("Fade out — switching to default pose mode")
                        self._set_default_pose_mode(True)
                        self._user_fade_out = True
                    elif not self._blend_out_active:
                        logger.info("Fade out — blending to default pose")
                        self._blend_out_active = True
                        self._blend_out_step = 0
                        self._user_fade_out = True
                        inner = self._inner_policy()
                        if hasattr(inner, "_paused"):
                            inner._paused = True

        self.ctrl_manager.post_step_callback(ctrl_data)

        self.policy.post_step_callback(commands)
        if self.visualizer is not None:
            self.policy.debug_viz(self.visualizer, env_data, ctrl_data, extras)

        self.safety_check()
        if self.cfg.debug.log_obs:
            self.debug_logger.log(
                env_data=env_data,
                ctrl_data=ctrl_data,
                extras=extras,
                pd_target=pd_target,
                timestep=self.timestep,
            )

    def step(self, dry_run=False):
        with self.recorder.measure("robot_state"):
            self.env.update()
            env_data = self.env.get_data()

        ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)

        commands = ctrl_data.get("COMMANDS", [])
        if len(commands) > 0:
            logger.info(f"{'=' * 10} COMMANDS {'=' * 10}\n{commands}")

        # -- WBC execution commands take effect THIS step (freeze/resume/
        #    damping/hands-ready/preview/confirm/single-step/burst/continuous) --
        if not dry_run:
            self.exec.handle_commands(commands, current_dof=self.env.dof_pos, ready_pose=self._ready_pose)
            if "[HANDS_READY]" in commands and self.hands is not None:
                self.hands.set_ready()

        with self.recorder.measure("policy_inference"):
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            pd_target = self.policy.get_pd_target(obs)

        # -- Detect motion done --
        callbacks = extras.get("CALLBACK", [])
        if "[MOTION_DONE]" in callbacks and not self._blend_out_active:
            if self._has_default_pose_mode:
                logger.info("Motion done — switching to default pose mode")
                self._set_default_pose_mode(True)
            else:
                logger.info("Motion done — blending out to default pose")
                self._blend_out_active = True
                self._blend_out_step = 0

        # -- Blend policy output → init pose (frame 0) --
        if self._blend_out_active:
            alpha = min(self._blend_out_step / max(self._blend_out_duration, 1), 1.0)
            pd_target = (1 - alpha) * pd_target + alpha * self._init_dof_pos
            self._blend_out_step += 1

        # -- WBC state machine resolves what actually reaches the robot --
        decision = self.exec.tick(pd_target, current_dof=self.env.dof_pos, timestep=self.timestep)
        send_target = decision.pd_target

        # Gate the policy frame advance when not in continuous RUNNING mode:
        # frozen/preview/damping hold the frame; stepped/confirmed advance once.
        inner = self._inner_policy()
        if self.exec.mode != ExecMode.RUNNING and hasattr(inner, "_paused"):
            inner._paused = not decision.advance_policy

        if not dry_run:
            with self.recorder.measure("command_send"):
                self.env.step(send_target, extras.get("hand_pose", None))

        self.post_step_callback(env_data, ctrl_data, extras, send_target)
        self.recorder.mark("env_step")

        # Handle pending blend-in (after MOTION_RESET / FADE_IN).
        if self._pending_blend_in:
            self._pending_blend_in = False
            self._run_blend_in()
            self._blend_in_completed = True

    def _run_blend_in(self):
        """Phase 2 of prepare: blend from default pose to policy output.

        Policy runs in default-pose mode (if supported); frame stays at 0
        (no post_step_callback).  After blend, switches to motion tracking.
        """
        secs = self._prepare_seconds or 3.0
        blend_steps = int(secs * self.freq)

        # Enter default-pose mode for the blend-in period.
        self._set_default_pose_mode(True)

        logger.warning(f"Blend-in: default DOF → policy ({blend_steps} steps, {secs:.1f}s)")
        pbar = ProgressBar("Blend in", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            action = (1 - alpha) * self._init_dof_pos + alpha * policy_pd

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Switch to motion tracking — policy sees the jump.
        self._set_default_pose_mode(False)
        logger.warning("Blend-in done — motion starting")

    def _teardown_deploy_resources(self):
        """Flush recorder + close hand interface on shutdown (idempotent)."""
        rec = getattr(self, "recorder", None)
        if rec is not None:
            try:
                rec.close()
            except Exception:
                pass
        hands = getattr(self, "hands", None)
        if hands is not None:
            try:
                hands.close()
            except Exception:
                pass

    def startup(self):
        """Deploy-default startup: ramp the robot to the READY pose with NO
        policy running, then HOLD frozen.

        Policy engagement is a deliberate, separate, user-triggered step
        ([RESUME_POLICY] / [POLICY_RUN_CONTINUOUS] / stepped commands).  This
        replaces the "immediately blend the policy in" behaviour on the deploy
        path so the operator confirms readiness before any policy action moves
        the hardware.
        """
        ready = self._ready_pose
        ramp_seconds = float(getattr(self.exec.cfg, "ramp_seconds", 2.0))
        ramp_steps = max(1, int(ramp_seconds * self.freq))

        logger.warning(
            f"startup: ramp to READY pose ({ramp_steps} steps, {ramp_seconds:.1f}s) "
            "— policy NOT running"
        )
        pbar = ProgressBar("Startup: ramp to ready", ramp_steps)

        last_step_time = time.time()
        for t in range(ramp_steps):
            current_motor_angle = np.array(self.env.dof_pos)
            alpha = min(t / max(ramp_steps - 1, 1), 1.0)
            action = (1 - alpha) * current_motor_angle + alpha * ready
            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Command the Psyonic hands to their ready pose too, if present.
        if self.hands is not None:
            self.hands.set_ready()

        # Latch the ready pose and hold FROZEN; wait for a deliberate engage.
        self.exec.force_freeze(ready, reason="startup ready pose")
        if hasattr(self._inner_policy(), "_paused"):
            self._inner_policy()._paused = True
        logger.warning(
            "startup done — at READY pose, policy FROZEN. Trigger [RESUME_POLICY] "
            "or a stepped command to engage the policy."
        )

    def prepare(self, init_motor_angle=None, prepare_seconds=None):
        if init_motor_angle is not None:
            desired_motor_angle = init_motor_angle
        elif self._has_default_pose_mode:
            # Ramp to the env's default standing pose (not motion frame 0).
            desired_motor_angle = np.array(
                self.env.dof_cfg.default_pos, dtype=np.float32
            )
        else:
            desired_motor_angle = self.policy.get_init_dof_pos()

        # Convert seconds to steps (at policy frequency).
        # Default: 3s ramp + 5s blend.  CLI --prepare-seconds overrides both.
        if prepare_seconds is not None:
            ramp_steps = int(prepare_seconds * self.freq)
            blend_steps = int(prepare_seconds * self.freq)
        else:
            ramp_steps = int(3.0 * self.freq)
            blend_steps = int(5.0 * self.freq)

        # ── Phase 1: Ramp joints to default pose ──
        logger.warning(
            f"prepare: phase 1 — ramp joints ({ramp_steps} steps, "
            f"{ramp_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: ramp joints", ramp_steps)

        last_step_time = time.time()
        for t in range(ramp_steps):
            current_motor_angle = np.array(self.env.dof_pos)
            alpha = min(t / max(ramp_steps - 1, 1), 1.0)
            action = (1 - alpha) * current_motor_angle + alpha * desired_motor_angle

            self.env.step(action)

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # Reset policy for a clean start — frame goes back to 0.
        self.reset()
        self._prepare_seconds = prepare_seconds  # restore after reset

        # ── Phase 2: Blend in policy (holding default pose) ──
        # Policy runs in default-pose mode (if supported): it sees synthetic
        # references for the standing pose, not the real motion.
        # Actions blend from raw default DOF to policy output.
        self._set_default_pose_mode(True)

        logger.warning(
            f"prepare: phase 2 — blend policy ({blend_steps} steps, "
            f"{blend_steps / self.freq:.1f}s)"
        )
        pbar = ProgressBar("Prepare: blend policy", blend_steps)

        last_step_time = time.time()
        for t in range(blend_steps):
            alpha = t / max(blend_steps - 1, 1)

            # Run policy observation + action (frame stays at 0).
            self.env.update()
            env_data = self.env.get_data()
            ctrl_data = self.ctrl_manager.get_ctrl_data(env_data)
            obs, extras = self.policy.get_observation(env_data, ctrl_data)
            policy_pd = self.policy.get_pd_target(obs)

            # Blend: default DOF → policy output
            action = (1 - alpha) * desired_motor_angle + alpha * policy_pd

            self.env.step(action)

            # Do NOT call post_step_callback — frame stays at 0.

            time_diff = last_step_time + self.dt - time.time()
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                logger.error("Warning: frame drop")
            last_step_time = time.time()
            pbar.update()
        pbar.close()

        # ── Phase 3: Hold default pose — wait for R to start motion ──
        # Stay in default-pose mode. Motion starts when [MOTION_RESET] is
        # received (user presses R), which calls _set_default_pose_mode(False).
        logger.warning("prepare done — holding default pose, press R to start motion")


if __name__ == "__main__":
    pass
