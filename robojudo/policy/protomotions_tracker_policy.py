"""ProtoMotions tracker policy for RoboJuDo.

Runs a unified ONNX model exported by
``deployment/export_bm_tracker_onnx.py`` with cached 50 fps motion from
``deployment/motion_utils.MotionPlayer``.

Key inputs:

- ``historical.processed_actions`` — action history feedback (previous PD
  targets are fed back as an ONNX input)
- ``mimic.future_anchor_rot`` — anchor-body-only rotation references

Heading alignment
-----------------
Yaw-only offset computed on first step to align motion heading with robot heading.

Sensor requirements (real G1)
-----------------------------
- ``env_data.dof_pos`` / ``env_data.dof_vel`` -- joint encoders
- ``env_data.base_quat`` (xyzw) -- pelvis IMU
- ``env_data.base_ang_vel`` -- pelvis IMU gyro (body-local frame)
- ``env_data.torso_quat`` (xyzw) -- FK-computed (requires ``update_with_fk=True``)
"""

import logging
import os
import re

import numpy as np
import onnxruntime as ort
import yaml

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig
from robojudo.utils.motion_utils import (
    LiveRefSource,
    MotionPlayer,
    _extract_yaw_quat_np,
    apply_heading_offset_np,
    compute_yaw_offset_np,
    quat_mul_np,
    quat_rotate_np,
)

logger = logging.getLogger(__name__)


@policy_registry.register
class ProtoMotionsTrackerPolicy(Policy):
    """Policy that drives a ProtoMotions tracker via unified ONNX model.

    The ONNX model bakes in: obs computation -> actor MLP -> action processing.
    Inputs are raw context tensors; outputs are absolute PD position targets.
    """

    cfg_policy: PolicyCfg

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        # Load YAML metadata BEFORE calling super().__init__ so we can
        # build the DOF config from it.
        onnx_path = cfg_policy.policy_file
        yaml_path = onnx_path.replace(".onnx", ".yaml")

        with open(yaml_path) as f:
            self._meta = yaml.safe_load(f)

        robot_meta = self._meta["robot"]
        control_meta = self._meta["control"]
        motion_meta = self._meta["motion"]
        runtime = self._meta["_runtime"]

        joint_names = robot_meta["joint_names"]
        num_dofs = robot_meta["num_dofs"]
        stiffness = control_meta["stiffness"]
        damping = control_meta["damping"]
        effort_limits = control_meta.get("effort_limits")

        # Build DOF config from YAML metadata.
        dof_cfg = DoFConfig(
            joint_names=joint_names,
            default_pos=[0.0] * num_dofs,
            stiffness=stiffness,
            damping=damping,
            torque_limits=effort_limits,
        )
        cfg_policy_updated = cfg_policy.model_copy()
        cfg_policy_updated.obs_dof = dof_cfg
        cfg_policy_updated.action_dof = dof_cfg

        super().__init__(cfg_policy=cfg_policy_updated, device="cpu")

        # ONNX session
        logger.info(f"[TrackerPolicy] Loading ONNX: {onnx_path}")
        self._session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self._onnx_in_names = [inp.name for inp in self._session.get_inputs()]
        self._onnx_out_names = [out.name for out in self._session.get_outputs()]
        self._onnx_name_to_key = runtime["onnx_name_to_in_key"]

        # Motion player (cached mode -- no protomotions import)
        motion_path = getattr(cfg_policy, "motion_path", None)
        if motion_path is None:
            raise ValueError("ProtoMotionsTrackerPolicyCfg must set motion_path")
        motion_index = getattr(cfg_policy, "motion_index", 0)
        timing = self._meta["timing"]
        self._player = MotionPlayer(
            motion_path, motion_index=motion_index, control_dt=timing["control_dt"]
        )

        # ONNX input config
        self._anchor_idx = robot_meta["anchor_body_index"]
        self._root_idx = robot_meta["root_body_index"]
        self._future_step_indices = motion_meta["future_step_indices"]

        # Determine how to read the anchor body rotation from env_data.
        # Use body name to look up in fk_info; pelvis uses base_quat directly.
        self._anchor_body_name = robot_meta.get("anchor_body_name")
        logger.info(
            f"[TrackerPolicy] anchor body: "
            f"{self._anchor_body_name or 'pelvis'} (idx={self._anchor_idx})"
        )

        # Action post-processing config
        self._pd_target_max_accel = control_meta.get("pd_target_max_accel")
        self._action_ema_alpha = control_meta.get("action_ema_alpha", 1.0)

        logger.info(
            f"[TrackerPolicy] {num_dofs} DOFs, "
            f"{self._player.total_frames} motion frames, "
            f"anchor_idx={self._anchor_idx}, root_idx={self._root_idx}"
        )

        # Resolve default standing pose from protomotions robot config.
        self._default_dof_pos = self._resolve_default_dof_pos(joint_names)

        # --- imprint teleop seam (additive) ---
        # By default the reference source IS the MotionPlayer (clip playback).
        # When ``teleop_ref`` is set, swap in a LiveRefSource that is fed by an
        # external teleop retargeter via ctrl_data.  The MotionPlayer still
        # loads (motion_path handling above is untouched); teleop only overrides
        # which object the policy reads references from.  Reversible: unset the
        # cfg flag and the policy behaves exactly as before.
        self._teleop_ref = bool(getattr(cfg_policy, "teleop_ref", False))
        if self._teleop_ref:
            self._ref_source = LiveRefSource(
                default_dof_pos=self._default_dof_pos,
                anchor_idx=self._anchor_idx,
                control_dt=timing["control_dt"],
            )
            logger.info("[TrackerPolicy] teleop_ref ON -- using LiveRefSource")
        else:
            self._ref_source = self._player
        # --- end imprint teleop seam ---

        self._heading_offset = None
        # Operator-as-odom commanded-root anchors (teleop_ref only; see the
        # displacement-command block in get_observation). Recaptured whenever
        # the heading offset is recaptured (per engage / reset), so unlimited
        # re-clutches restart the commanded trajectory at the robot's
        # then-current anchor with zero error.
        self._cmd_ref_p0 = None
        self._cmd_robot_p0 = None
        self._cmd_offset = None
        self.reset()

    # Default standing pose per robot (from protomotions.robot_configs.*).
    # NOTE: h1_2 entry is an APPROXIMATION reusing g1's regex/values since
    # h1_2's joint-name suffixes match g1's convention (hip_pitch/knee/
    # ankle_pitch/elbow/shoulder_roll+pitch) and no explicit default standing
    # joint-angle table was found in protomotions/robot_configs/h1_2.py at
    # smoke-checkpoint time. Only used for the "hold default pose" UX mode
    # (pre-motion-start); does not affect tracker rollout correctness once
    # motion tracking starts. Revisit with real h1_2-tuned values if the
    # default-pose hold looks bad.
    _DEFAULT_JOINT_POS_BY_ROBOT = {
        "g1": {
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        "h1_2": {
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
    }
    # Backward-compat alias (in case anything external referenced this name).
    _G1_DEFAULT_JOINT_POS = _DEFAULT_JOINT_POS_BY_ROBOT["g1"]

    def _resolve_default_dof_pos(self, joint_names: list[str]) -> np.ndarray:
        """Resolve default DOF positions via regex-pattern matching."""
        robot = getattr(self.cfg_policy, "robot", "g1")
        DEFAULT_JOINT_POS = self._DEFAULT_JOINT_POS_BY_ROBOT.get(
            robot, self._DEFAULT_JOINT_POS_BY_ROBOT["g1"]
        )

        default_pos = np.zeros(len(joint_names), dtype=np.float32)
        for pattern, value in DEFAULT_JOINT_POS.items():
            for i, name in enumerate(joint_names):
                if re.fullmatch(pattern, name):
                    default_pos[i] = value
        logger.info(f"[TrackerPolicy] resolved default DOF pos: {default_pos}")
        return default_pos

    def set_default_pose_mode(self, enabled: bool):
        """Switch between tracking real motion and holding default pose.

        When enabled, the policy sees synthetic references for the default
        standing pose instead of the real motion.  Used during prepare/rampdown.
        """
        self._default_pose_mode = enabled
        if enabled:
            self._motion_done = False
        logger.info(f"[TrackerPolicy] default_pose_mode={'ON' if enabled else 'OFF'}")

    def reset(self):
        self._frame = 0
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev = None
        self._stashed_pd_targets = np.zeros(self.num_actions, dtype=np.float32)
        self._prev_actions = np.zeros(self.num_actions, dtype=np.float32)
        self._motion_done = False
        self._paused = False
        self._default_pose_mode = False

    def reset_alignment(self):
        self._heading_offset = None
        self._cmd_ref_p0 = None

    def post_step_callback(self, commands=None):
        if not self._paused and not self._default_pose_mode:
            self._frame += 1
            if self._frame >= self._ref_source.total_frames:
                self._frame = self._ref_source.total_frames - 1
                self._motion_done = True
        for cmd in commands or []:
            if cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()

    def get_observation(self, env_data, ctrl_data):
        # --- imprint teleop seam (additive) ---
        # Pump the latest retargeted reference from ctrl_data into the live
        # source before it is read below.  ctrl_data is a Box; the TeleopCtrl
        # payload lives under the "TeleopCtrl" key.  Missing/None arrays are
        # skipped (the LiveRefSource keeps its last / seeded default value).
        # _paused must also freeze the LIVE reference: for a MotionPlayer,
        # pausing the frame counter (post_step_callback) is enough, but the
        # LiveRefSource ignores the frame argument and serves whatever was
        # last update()d -- so keep pumping while paused and the robot keeps
        # following the operator with "freeze ON" in the log. Skipping the
        # update leaves the last buffered reference held, which IS the freeze.
        if self._teleop_ref and not self._paused:
            # The payload key is the CONTROLLER's ctrl_type: upstream
            # RoboJuDo's TeleopCtrl publishes under "TeleopCtrl", imprint's
            # ImprintTeleopCtrl (the gated lane since it moved off the
            # upstream cfg) under "ImprintTeleopCtrl". Reading only the
            # former silently starved this pump in the gated lane -- the
            # LiveRefSource held its seeded standing default forever and the
            # robot ignored teleop while everything upstream looked healthy.
            teleop = {}
            if ctrl_data is not None:
                teleop = (
                    ctrl_data.get("TeleopCtrl")
                    or ctrl_data.get("ImprintTeleopCtrl")
                    or {}
                )
            # Per-engage heading re-arm, ordering-proof: the provider stamps
            # the engage generation its reference is consistent with; a change
            # means the operator re-clutched, so the yaw offset captured
            # against the previous engage's anchor row is stale. Clearing it
            # here -- BEFORE the pump and the lazy recapture below -- makes
            # the recapture see this tick's fresh anchor row and the robot's
            # current yaw, so the relative heading command restarts at zero
            # (no lurch) on EVERY engage, not just the first. The
            # gated_inference engage watcher does the same thing when its
            # registration wins the race with the first align; this is the
            # authoritative path. Idempotent with it.
            gen = teleop.get("engage_generation", None)
            if gen is not None and teleop.get("ref_anchor_pos", None) is not None:
                # Scoped to operator-as-odom (ref_anchor_pos present): the
                # default lane keeps its existing lazy capture + watcher
                # behaviour, byte-identical.
                last_gen = getattr(self, "_teleop_engage_generation", None)
                if last_gen is None or gen != last_gen:
                    # Also fires on the FIRST odom tick (last_gen None): any
                    # offset captured during the pre-engage default-pose hold
                    # was measured against the seeded identity row, not the
                    # odom row, and must not survive into tracking.
                    self._heading_offset = None
                self._teleop_engage_generation = gen
            ref_dof_pos = teleop.get("ref_dof_pos", None)
            ref_dof_vel = teleop.get("ref_dof_vel", None)
            ref_body_rot = teleop.get("ref_body_rot", None)
            # Optional operator-as-odom position channel (imprint teleop):
            # a stitched-continuous operator torso trajectory. When present,
            # the LiveRefSource grows a body_pos anchor channel and the
            # anchor-position displacement command below becomes live instead
            # of falling back to zero. None -> previous behaviour.
            ref_anchor_pos = teleop.get("ref_anchor_pos", None)
            if (
                ref_dof_pos is not None
                and ref_dof_vel is not None
                and ref_body_rot is not None
            ):
                self._ref_source.update(
                    ref_dof_pos, ref_dof_vel, ref_body_rot,
                    anchor_pos=ref_anchor_pos,
                    engage_generation=gen,
                )
        # --- end imprint teleop seam ---

        # -- Heading alignment (first step after reset) --
        heading_recaptured = False
        if self._heading_offset is None:
            motion_anchor_rot = self._ref_source.get_state_at_frame(0)["body_rot"][self._anchor_idx]
            robot_anchor_rot = self._get_anchor_quat(env_data)
            self._heading_offset = compute_yaw_offset_np(robot_anchor_rot, motion_anchor_rot)
            heading_recaptured = True

        # -- State from env_data (already xyzw) --
        anchor_rot = self._get_anchor_quat(env_data)
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        # env_data.base_ang_vel comes from MuJoCo qvel[3:6] which is ALREADY
        # in the pelvis local frame (not world frame).  On the real G1, the
        # IMU gyroscope also reads in body-local frame.  So we use it directly
        # as root_local_ang_vel -- NO quat_rotate_inverse needed.
        root_local_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)

        anchor_pos = self._get_anchor_pos(env_data)

        if self._default_pose_mode:
            # -- Synthetic references: hold default standing pose --
            # Target DOFs = default standing pose, velocities = zero,
            # anchor rotation = yaw-only from robot's current anchor (hold
            # heading but neutral pitch/roll for stable upright standing).
            num_steps = len(self._future_step_indices)
            anchor_yaw_only = _extract_yaw_quat_np(anchor_rot)
            future_anchor_rot = np.tile(anchor_yaw_only, (num_steps, 1))
            future_dof_pos = np.tile(self._default_dof_pos, (num_steps, 1))
            future_dof_vel = np.zeros_like(future_dof_pos)
            # No displacement command while holding pose -- "stay put".
            future_anchor_pos = np.tile(anchor_pos, (num_steps, 1))
        else:
            # -- Future motion references with heading alignment --
            # Clamp each future step so it never exceeds the last valid frame.
            # This repeats the last frame's references at end-of-motion instead
            # of going out of bounds.
            last_frame = self._ref_source.total_frames - 1
            clamped_steps = [min(self._frame + step, last_frame) - self._frame for step in self._future_step_indices]
            future_refs = self._ref_source.get_future_references(self._frame, clamped_steps)
            future_body_rot = apply_heading_offset_np(self._heading_offset, future_refs["body_rot"])
            # Anchor-body-only rotation: [num_steps, 4]
            future_anchor_rot = future_body_rot[:, self._anchor_idx, :]
            future_dof_pos = future_refs["dof_pos"]
            future_dof_vel = future_refs["dof_vel"]

            # -- Anchor position displacement command --
            # Some (dense DeepMimic-style) teachers additionally condition on
            # a heading-local XYZ displacement command: reference-anchor
            # motion minus current-anchor motion, rotated into the robot's
            # current heading frame (see
            # protomotions/envs/obs/mimic_command.py::build_mimic_future_displacement_cmd).
            # It is a pure delta (translation-invariant), so we don't need the
            # robot and the motion clip to share a world frame: we compute the
            # motion's own raw anchor-position delta, re-express it in the
            # robot's heading-aligned frame (the SAME `self._heading_offset`
            # used for orientation above, composed with the robot's current
            # yaw since the exported obs kernel rotates by the fed
            # `current.anchor_rot`'s yaw internally), and add it on top of the
            # robot's actual current anchor position (which is otherwise an
            # arbitrary/unused origin -- only the delta matters).
            ref_body_pos = future_refs.get("body_pos")
            cur_motion_state = self._ref_source.get_state_at_frame(self._frame)
            cur_body_pos = cur_motion_state.get("body_pos")
            if ref_body_pos is not None and cur_body_pos is not None and self._teleop_ref:
                # Operator-as-odom (teleop_ref + a live position channel):
                # feed an ABSOLUTE commanded root trajectory in the robot's
                # world frame, anchored per engage:
                #
                #   cmd(t+k) = robot_p@capture
                #            + R(heading_offset@capture) * (op_p(t+k) - op_p@capture)
                #
                # so the ONNX obs kernel's displacement (future - current
                # anchor, rotated into the robot's live heading) reproduces
                # the TRAINING semantics: ref_future - robot_now, i.e. the
                # robot's own tracking error feeds back and it catches up to
                # the commanded trajectory instead of open-loop-following a
                # feedforward velocity (the previous ref-delta feed, which
                # walked at ~25% of the commanded speed because the error
                # term never accumulated). XY only: the Z channel stays the
                # previous feedforward ref-delta on top of the robot's
                # current height (odom commands XY+heading, never an
                # absolute height -- operator and robot torso heights don't
                # share a scale).
                cur_ref_p = cur_body_pos[self._anchor_idx]
                if heading_recaptured or self._cmd_ref_p0 is None:
                    self._cmd_ref_p0 = cur_ref_p.copy()
                    self._cmd_robot_p0 = anchor_pos.copy()
                    self._cmd_offset = self._heading_offset.copy()
                ref_delta = ref_body_pos[:, self._anchor_idx, :] - self._cmd_ref_p0
                cmd = self._cmd_robot_p0[None, :] + quat_rotate_np(
                    self._cmd_offset, ref_delta
                )
                # Z: feedforward delta over the horizon (previous behaviour).
                cmd[:, 2] = anchor_pos[2] + (
                    ref_body_pos[:, self._anchor_idx, 2]
                    - cur_body_pos[self._anchor_idx, 2]
                )
                # Bound the XY displacement obs the policy will see. Training
                # never shows multi-metre displacements (max ~ref speed *
                # 0.4 s horizon plus modest tracking error); a reference
                # discontinuity (e.g. a looped tape, a dropped stream) must
                # degrade to a max-speed walk command, not an
                # out-of-distribution sprint.
                disp_xy = cmd[:, :2] - anchor_pos[None, :2]
                norms = np.linalg.norm(disp_xy, axis=-1, keepdims=True)
                max_disp = 1.0
                scale = np.where(norms > max_disp, max_disp / np.maximum(norms, 1e-9), 1.0)
                cmd[:, :2] = anchor_pos[None, :2] + disp_xy * scale
                future_anchor_pos = cmd
            elif ref_body_pos is not None and cur_body_pos is not None:
                motion_delta = (
                    ref_body_pos[:, self._anchor_idx, :] - cur_body_pos[self._anchor_idx]
                )
                robot_yaw_now = _extract_yaw_quat_np(anchor_rot)
                compose_quat = quat_mul_np(robot_yaw_now, self._heading_offset)
                delta_in_robot_frame = quat_rotate_np(compose_quat, motion_delta)
                future_anchor_pos = anchor_pos[None, :] + delta_in_robot_frame
            else:
                # Reference source has no position channel (e.g. teleop's
                # LiveRefSource, which only tracks dof_pos/dof_vel/body_rot).
                # Fall back to a zero displacement command.
                future_anchor_pos = np.tile(anchor_pos, (future_anchor_rot.shape[0], 1))

        # --- imprint ghost seam (additive) ---
        # Stamp the reference pose for THIS tick (frame t, the state the
        # tracking metrics pair with rollout tick t) plus the commanded root,
        # for the translucent reference ghost the sim lanes draw
        # (imprint.robojudo.teleop.ghost.install_env_ghost). Pure annotation:
        # nothing below reads it. Commanded root XY reproduces the odom
        # displacement command at k=0 (same anchors as the future_anchor_pos
        # block above); without odom it is the robot's own anchor (the
        # command is "stay put"). Yaw is the heading-aligned reference anchor
        # yaw -- exactly what future_anchor_rot feeds the ONNX.
        try:
            if self._default_pose_mode:
                _g_dof = self._default_dof_pos
                _g_pos = anchor_pos
                _g_rot = _extract_yaw_quat_np(anchor_rot)
            else:
                _g_cur = self._ref_source.get_state_at_frame(self._frame)
                _g_dof = _g_cur["dof_pos"]
                _g_rot = apply_heading_offset_np(
                    self._heading_offset, _g_cur["body_rot"][None]
                )[0][self._anchor_idx]
                _g_pos = anchor_pos
                _g_cbp = _g_cur.get("body_pos")
                if (self._teleop_ref and self._cmd_ref_p0 is not None
                        and _g_cbp is not None):
                    _g_d = _g_cbp[self._anchor_idx] - self._cmd_ref_p0
                    _g_p = self._cmd_robot_p0 + quat_rotate_np(self._cmd_offset, _g_d)
                    _g_pos = np.array([_g_p[0], _g_p[1], anchor_pos[2]],
                                      dtype=np.float32)
            self.last_ghost_reference = {
                "dof_pos": np.asarray(_g_dof, dtype=np.float32).copy(),
                "root_pos": np.asarray(_g_pos, dtype=np.float32).copy(),
                # xyzw quat -> yaw
                "root_yaw": float(2.0 * np.arctan2(_g_rot[2], _g_rot[3])),
            }
        except Exception:  # noqa: BLE001 -- the ghost must never break tracking
            self.last_ghost_reference = None
        # --- end imprint ghost seam ---

        if os.environ.get("IMPRINT_ODOM_DEBUG"):
            self._dbg_n = getattr(self, "_dbg_n", 0) + 1
            if self._dbg_n % 25 == 1:
                _teleop_keys = None
                if self._teleop_ref and ctrl_data is not None:
                    _t = (
                        ctrl_data.get("TeleopCtrl")
                        or ctrl_data.get("ImprintTeleopCtrl")
                        or {}
                    )
                    _teleop_keys = {
                        "ref_anchor_pos": _t.get("ref_anchor_pos", None),
                        "gen": _t.get("engage_generation", None),
                    }
                _disp = future_anchor_pos - anchor_pos[None, :]
                _hist = len(getattr(self._ref_source, "_history", []))
                _vel = getattr(self._ref_source, "_anchor_vel", None)
                print(
                    f"[odom-dbg] n={self._dbg_n} default_pose={self._default_pose_mode} "
                    f"teleop={_teleop_keys} hist_len={_hist} ema_vel={_vel} "
                    f"disp_norms={[round(float(np.linalg.norm(d[:2])), 4) for d in _disp]} "
                    f"disp_step4_xy={_disp[3][:2] if _disp.shape[0] > 3 else None}",
                    flush=True,
                )

        # -- Build ONNX inputs --
        key_to_array = {
            "current.dof_pos": dof_pos[None],
            "current.dof_vel": dof_vel[None],
            "current.anchor_rot": anchor_rot[None],
            "current.anchor_pos": anchor_pos[None],
            "current.root_local_ang_vel": root_local_ang_vel[None],
            "mimic.future_anchor_rot": future_anchor_rot[None],
            "mimic.future_anchor_pos": future_anchor_pos[None],
            "mimic.future_dof_pos": future_dof_pos[None],
            "mimic.future_dof_vel": future_dof_vel[None],
            "historical.processed_actions": self._prev_actions[None, None],
        }
        # noisy.*: ProtoMotions' observation-noise view of the robot state.
        # Noise is training-only DR; at inference the noisy view aliases the
        # clean tensors, so checkpoints exported with noisy_* input bindings
        # (e.g. Track D teachers) are fed the same state arrays.
        for _clean in ("dof_pos", "dof_vel", "anchor_rot", "anchor_pos", "root_local_ang_vel"):
            key_to_array["noisy." + _clean] = key_to_array["current." + _clean]
        onnx_inputs = {}
        for onnx_name in self._onnx_in_names:
            sem_key = self._onnx_name_to_key.get(onnx_name)
            if sem_key and sem_key in key_to_array:
                onnx_inputs[onnx_name] = key_to_array[sem_key].astype(np.float32)

        # -- ONNX inference --
        ort_out = self._session.run(self._onnx_out_names, onnx_inputs)
        pd_targets = ort_out[1].squeeze().copy()

        # -- PD target acceleration clamp --
        if (
            self._pd_target_max_accel is not None
            and self._prev_pd is not None
            and self._prev_prev_pd is not None
        ):
            delta = pd_targets - self._prev_pd
            prev_delta = self._prev_pd - self._prev_prev_pd
            accel = delta - prev_delta
            clamped_accel = np.clip(
                accel, -self._pd_target_max_accel, self._pd_target_max_accel
            )
            pd_targets = self._prev_pd + prev_delta + clamped_accel
        self._prev_prev_pd = self._prev_pd
        self._prev_pd = pd_targets.copy()

        # -- EMA action filter --
        alpha = self._action_ema_alpha
        if alpha < 1.0:
            if self._ema_prev is None:
                self._ema_prev = pd_targets.copy()
            pd_targets = alpha * pd_targets + (1.0 - alpha) * self._ema_prev
            self._ema_prev = pd_targets.copy()

        self._stashed_pd_targets = pd_targets
        self._prev_actions = pd_targets.copy()
        extras = {
            "CALLBACK": (
                ["[MOTION_DONE]"]
                if self._motion_done and not self._default_pose_mode
                else []
            ),
        }
        dummy_obs = np.zeros(1, dtype=np.float32)
        return dummy_obs, extras

    def _get_anchor_quat(self, env_data) -> np.ndarray:
        """Read the anchor body's quaternion from env_data.

        Uses the body name from YAML metadata to look up in fk_info.
        Falls back to base_quat for pelvis (root body).
        """
        name = self._anchor_body_name
        if name is not None and name not in (None, "pelvis"):
            # Named body -- look up in FK info
            fk = env_data.fk_info
            if fk is not None and name in fk:
                return np.asarray(fk[name]["quat"], dtype=np.float32)
            # Fallback: if the env exposes it as torso_quat and name matches
            if name == "torso_link" and env_data.torso_quat is not None:
                return np.asarray(env_data.torso_quat, dtype=np.float32)
        # Pelvis / root body -- always available as base_quat
        return np.asarray(env_data.base_quat, dtype=np.float32)

    def _get_anchor_pos(self, env_data) -> np.ndarray:
        """Read the anchor body's world position from env_data.

        Mirrors :meth:`_get_anchor_quat`. Only used as a translation-invariant
        origin for the anchor-position displacement command (see
        ``get_observation``) -- its absolute value is never meaningful on its
        own, only the delta between "current" and "future" anchor positions.
        """
        name = self._anchor_body_name
        if name is not None and name not in (None, "pelvis"):
            fk = env_data.fk_info
            if fk is not None and name in fk:
                return np.asarray(fk[name]["pos"], dtype=np.float32)
            if name == "torso_link" and getattr(env_data, "torso_pos", None) is not None:
                return np.asarray(env_data.torso_pos, dtype=np.float32)
        return np.asarray(env_data.base_pos, dtype=np.float32)

    def get_action(self, obs):
        return self._stashed_pd_targets

    def get_init_dof_pos(self):
        return self._ref_source.get_state_at_frame(0)["dof_pos"].copy()
