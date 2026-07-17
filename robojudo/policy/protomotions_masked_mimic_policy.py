"""ProtoMotions MaskedMimic (Track C) sparse-teleop policy for RoboJuDo.

Runs a unified ONNX model exported by
``deployment/export_masked_mimic_onnx.py`` (mean-latent VAE student: prior
transformer -> prior_mu -> trunk MLP -> actions). The graph bakes the full
obs computation, so this policy only assembles RAW context tensors at 50 Hz
in numpy:

- full-body max-coords robot state via FK (``robojudo.tools.kinematics
  .MujocoKinematics`` on the checkpoint's own MJCF): world body pos/rot
  (xyzw) + world lin vel + world ang vel (MuJoCo ``cvel`` convention --
  identical to ProtoMotions' MuJoCo simulator state extraction);
- a 5-step historical rigid-body ring buffer (index 0 = t-dt, matching
  ProtoMotions' ``StateHistoryBuffer`` semantics where the current state
  occupies buffer slot 0 and the historical view starts at slot 1);
- a raw-action history buffer (the model's ``previous_actions`` obs is the
  action from TWO policy queries ago -- the training-time buffer semantics;
  verified against the stored-input fixture);
- sparse conditioning targets (3point = head + wrists, 5point = + ankles)
  from the live teleop seam (``ctrl_data["TeleopCtrl"]["world_targets"]``,
  z-up world, measured-normalized by the zmq smpl source), registered to the
  robot on tracking start (per-body offset so targets meet the robot where
  it stands -- the same frame0 semantics as
  ``imprint.protomotions.masked_mimic_rollout --live``);
- beta-sampled target times with a receding horizon (the trained interface:
  ``Beta(time_alpha, time_beta)`` offsets, resampled whenever the earliest
  target time expires -- exactly ``MaskedMimicControl``'s machinery with the
  remaining-clip-length replaced by a fixed look-ahead horizon).

Sensor requirements: ``dof_pos``/``dof_vel``, ``base_pos``, ``base_quat``
(xyzw), ``base_ang_vel`` (body-local), ``base_lin_vel`` (world; None on real
hardware -> zeros, a documented sim2real gap).

``_paused`` freezes BOTH the virtual clock and the live-target pump (holding
the last injected targets) -- the live-freeze fix pattern from
``ProtoMotionsTrackerPolicy``.
"""

import logging
import re
import time

import numpy as np
import onnxruntime as ort
import yaml

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.kinematics import MujocoKinematics
from robojudo.tools.tool_cfgs import DoFConfig, ForwardKinematicCfg

logger = logging.getLogger(__name__)


def _yaw_of_quat_xyzw(q) -> float:
    """Heading yaw of an xyzw quaternion: yaw of the rotated +x (forward) axis."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    fx = 1.0 - 2.0 * (y * y + z * z)  # (R @ [1,0,0]).x
    fy = 2.0 * (x * y + w * z)        # (R @ [1,0,0]).y
    return float(np.arctan2(fy, fx))


def _rot_z(yaw: float) -> np.ndarray:
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# ctrl_data world-target keys -> ProtoMotions body names (see
# imprint.robojudo.teleop.sources.zmq_source._DENSE_KEYS / targets_from_smpl).
WORLD_KEY_TO_BODY = {
    "head": "head_aux",
    "left_hand": "left_wrist_yaw_link",
    "right_hand": "right_wrist_yaw_link",
    "left_foot": "left_ankle_roll_link",
    "right_foot": "right_ankle_roll_link",
}
CONDITIONING_BODIES = {
    "3point": ("head_aux", "left_wrist_yaw_link", "right_wrist_yaw_link"),
    "5point": (
        "head_aux",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ),
}


class MaskedMimicStateBuilder:
    """FK: (dof_pos, root state) -> full-body max-coords arrays.

    Body state extraction matches ProtoMotions' MuJoCo simulator exactly:
    ``xpos`` / ``xquat``(->xyzw) / ``cvel[3:6]`` (world lin) / ``cvel[0:3]``
    (world ang). Kept as a standalone class so the stored-input bisect
    (scratchpad/mm_onnx_bisect.py, part C) can validate it in isolation.
    """

    def __init__(self, mjcf_path: str, body_names: list, joint_names: list):
        self._kin = MujocoKinematics(
            ForwardKinematicCfg(xml_path=mjcf_path, kinematic_joint_names=list(joint_names))
        )
        self.body_names = list(body_names)
        missing = [n for n in self.body_names if n not in self._kin.body_names]
        if missing:
            raise ValueError(
                f"MJCF {mjcf_path} lacks bodies {missing} (has {self._kin.body_names})"
            )

    def body_state(
        self,
        dof_pos: np.ndarray,
        dof_vel: np.ndarray,
        base_pos: np.ndarray,
        base_quat_xyzw: np.ndarray,
        base_lin_vel: np.ndarray | None,
        base_ang_vel: np.ndarray | None,
    ) -> dict:
        fk = self._kin.forward(
            joint_pos=np.asarray(dof_pos, dtype=np.float64),
            base_pos=np.asarray(base_pos, dtype=np.float64),
            base_quat=np.asarray(base_quat_xyzw, dtype=np.float64),
            joint_vel=np.asarray(dof_vel, dtype=np.float64),
            base_lin_vel=None if base_lin_vel is None else np.asarray(base_lin_vel, dtype=np.float64),
            base_ang_vel=None if base_ang_vel is None else np.asarray(base_ang_vel, dtype=np.float64),
        )
        n = len(self.body_names)
        pos = np.zeros((n, 3), dtype=np.float32)
        rot = np.zeros((n, 4), dtype=np.float32)
        vel = np.zeros((n, 3), dtype=np.float32)
        ang = np.zeros((n, 3), dtype=np.float32)
        for i, name in enumerate(self.body_names):
            info = fk[name]
            pos[i] = info["pos"]
            rot[i] = info["quat"]  # already xyzw (kinematics.py converts)
            vel[i] = info["lin_vel"]
            ang[i] = info["ang_vel"]
        return {"pos": pos, "rot": rot, "vel": vel, "ang_vel": ang}


@policy_registry.register
class ProtoMotionsMaskedMimicPolicy(Policy):
    """Sparse-teleop masked-mimic student via unified ONNX model."""

    cfg_policy: PolicyCfg

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        onnx_path = cfg_policy.policy_file
        yaml_path = onnx_path.replace(".onnx", ".yaml")
        with open(yaml_path) as f:
            self._meta = yaml.safe_load(f)

        robot_meta = self._meta["robot"]
        control_meta = self._meta["control"]
        mm_meta = self._meta["masked_mimic"]
        runtime = self._meta["_runtime"]
        timing = self._meta["timing"]

        joint_names = robot_meta["joint_names"]
        num_dofs = robot_meta["num_dofs"]
        dof_cfg = DoFConfig(
            joint_names=joint_names,
            default_pos=[0.0] * num_dofs,
            stiffness=control_meta["stiffness"],
            damping=control_meta["damping"],
            torque_limits=control_meta.get("effort_limits"),
        )
        cfg_updated = cfg_policy.model_copy()
        cfg_updated.obs_dof = dof_cfg
        cfg_updated.action_dof = dof_cfg
        super().__init__(cfg_policy=cfg_updated, device="cpu")

        logger.info(f"[MaskedMimicPolicy] Loading ONNX: {onnx_path}")
        # Cap ORT intra-op threads: the default (all cores) oversubscribes on a
        # busy control box and doubles p99 (measured 19ms -> 7ms at intra=4 for
        # this prior-transformer graph). 4 threads is the sweet spot at 50 Hz.
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"]
        )
        self._onnx_in_names = [i.name for i in self._session.get_inputs()]
        self._onnx_out_names = [o.name for o in self._session.get_outputs()]
        self._onnx_name_to_key = runtime["onnx_name_to_in_key"]

        self._body_names = list(robot_meta["body_names"])
        self._num_bodies = len(self._body_names)
        self._num_contact = int(robot_meta.get("num_contact_bodies", 2))
        self._control_dt = float(timing["control_dt"])
        self._S = int(mm_meta["num_masked_future_steps"])
        self._H = int(mm_meta["num_history_steps"])
        self._time_alpha = float(mm_meta["time_alpha"])
        self._time_beta = float(mm_meta["time_beta"])
        self._trackable = list(mm_meta["trackable_bodies_subset"])
        self._cond_lookahead = float(getattr(cfg_policy, "target_horizon_s", 4.0))

        conditioning = getattr(cfg_policy, "conditioning", "3point")
        self._cond_bodies = CONDITIONING_BODIES[conditioning]
        self._cond_body_ids = [self._body_names.index(n) for n in self._cond_bodies]
        # masks are in trackable-subset order: (S, num_trackable, 2) trans/rot
        masks = np.zeros((self._S, len(self._trackable), 2), dtype=np.float32)
        for name in self._cond_bodies:
            masks[:, self._trackable.index(name), 0] = 1.0  # translation only
        self._target_bodies_masks = masks.reshape(1, -1)
        self._target_poses_masks = np.ones((1, self._S), dtype=np.float32)
        logger.info(
            f"[MaskedMimicPolicy] conditioning={conditioning} on {self._cond_bodies} "
            f"(translation-only, always visible), horizon={self._cond_lookahead}s"
        )

        self._builder = MaskedMimicStateBuilder(
            mjcf_path=robot_meta["mjcf_path"],
            body_names=self._body_names,
            joint_names=joint_names,
        )

        self._pd_target_max_accel = control_meta.get("pd_target_max_accel")
        self._action_ema_alpha = control_meta.get("action_ema_alpha", 1.0)
        self._default_dof_pos = self._resolve_default_dof_pos(joint_names)

        # Optional init-pose motion (same seam as the tracker; gated_inference
        # always provides one). Only used for get_init_dof_pos().
        self._init_pose = None
        motion_path = getattr(cfg_policy, "motion_path", None)
        if motion_path:
            try:
                from robojudo.utils.motion_utils import MotionPlayer

                player = MotionPlayer(
                    motion_path,
                    motion_index=getattr(cfg_policy, "motion_index", 0),
                    control_dt=self._control_dt,
                )
                self._init_pose = player.get_state_at_frame(0)["dof_pos"].copy()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[MaskedMimicPolicy] init-pose motion load failed: {exc}")

        self._teleop_ref = bool(getattr(cfg_policy, "teleop_ref", False))
        self._rng = np.random.default_rng(0)
        self._latency_ms: list = []
        self.reset()

    # Same default standing pose table as ProtoMotionsTrackerPolicy.
    _DEFAULT_JOINT_POS = {
        ".*_hip_pitch_joint": -0.312,
        ".*_knee_joint": 0.669,
        ".*_ankle_pitch_joint": -0.363,
        ".*_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }

    def _resolve_default_dof_pos(self, joint_names):
        out = np.zeros(len(joint_names), dtype=np.float32)
        for pattern, value in self._DEFAULT_JOINT_POS.items():
            for i, name in enumerate(joint_names):
                if re.fullmatch(pattern, name):
                    out[i] = value
        return out

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        self._prev_pd = None
        self._prev_prev_pd = None
        self._ema_prev = None
        self._stashed_pd_targets = np.zeros(self.num_actions, dtype=np.float32)
        # Raw-action buffer, most recent FIRST; slot 0 = a_{t-1}, slot 1 =
        # a_{t-2} (what the model sees as previous_actions), ... H+1 slots.
        self._action_buf = np.zeros((self._H + 1, self.num_actions), dtype=np.float32)
        self._state_hist: list = []  # most recent first, states at t-dt..t-H*dt
        self._paused = False
        self._default_pose_mode = False
        self._motion_done = False
        # Compatibility shims for gated_inference's tracker-shaped pokes
        # (_stash_live_teleop_ref_source / _set_motion_frame_hold read these).
        self._ref_source = None
        self._frame = 0
        self._register_offset = None   # {body: (3,)} robot0 - live0 (frame0 delta)
        self._register_live0 = None    # {body: (3,)} live targets at registration
        self._register_robot0 = None   # {body: (3,)} FK body pos at registration
        self._register_R = None        # (3,3) Rz(robot_yaw - operator_yaw)
        self._hold_targets = None      # {body: (3,)} captured stand targets
        self._live_targets = None      # {body: (3,)} latest injected targets
        self._live_root_quat = None    # (4,) xyzw operator root orientation
        # XR-recenter guard: a >1 m single-tick target jump mid-tracking
        # freezes the injected targets until the next (re-)registration.
        self._recenter_frozen = False
        self._recenter_warned = 0.0
        # Re-registration target blend (see reset_alignment): blend the
        # INJECTED TARGETS from their last value to the newly registered ones
        # over ~0.5 s. Target-space (not offset-space) so a clutch resume --
        # operator moved while frozen, offsets change by that motion -- is
        # continuous by construction.
        self._target_blend_from = None   # {body: (3,)} last injected targets
        self._target_blend_left = 0
        self._target_blend_steps = max(int(round(0.5 / self._control_dt)), 1)
        self._last_injected = None       # {body: (3,)} updated every tick
        self._was_paused = False
        # Tracking gate (state-machine contract, matching the dense tracker):
        # OFF (default) = armed/READY-HOLD/post-damp -- the policy conditions
        # on its captured current-stance HOLD targets and ignores the live
        # world_targets pump entirely. ON (go/'r'/X via gated_inference's
        # set_tracking_active hook) = register frame0 offsets against the
        # current stance and start consuming live targets.
        self._tracking_active = False
        # Virtual reference clock + beta-sampled target times.
        self._t = 0.0
        self._target_times = None

    def reset_alignment(self):
        """Re-register live targets against the robot's current stance.

        Does NOT flip the tracking gate -- a second 'r' while tracking
        re-anchors the targets; going live from armed/hold is
        :meth:`set_tracking_active`. A re-register WHILE tracking blends the
        old registration offset into the new one over ~0.5 s so the target
        (and hence the action) doesn't step by the instantaneous tracking
        error.
        """
        if self._tracking_active and self._last_injected is not None:
            self._target_blend_from = {
                b: t.copy() for b, t in self._last_injected.items()
            }
            self._target_blend_left = self._target_blend_steps
        self._clear_registration()
        self._hold_targets = None

    def _clear_registration(self):
        self._register_offset = None
        self._register_live0 = None
        self._register_robot0 = None
        self._register_R = None
        self._recenter_frozen = False
        # Drop the cached live frame too: after an XR-recenter freeze the next
        # accepted frame would otherwise be compared against the stale
        # pre-jump one and re-trip the guard (the pump refills this on the
        # same tick the re-registration lands).
        self._live_targets = None

    def set_tracking_active(self, active: bool):
        """Gate live-target tracking (the tracker state-machine contract).

        ON: drop the hold targets and force a fresh frame0 registration on
        the next tick (targets meet the robot where it stands -- zero jump).
        OFF: back to hold mode -- the current stance is re-captured on the
        next tick and live targets are ignored until the next ON.
        """
        active = bool(active)
        if active == self._tracking_active:
            return
        self._tracking_active = active
        self._clear_registration()
        self._hold_targets = None
        self._target_blend_from = None
        self._target_blend_left = 0
        if not active:
            self._live_targets = None
        logger.info(
            f"[MaskedMimicPolicy] tracking {'ACTIVE (registering on next tick)' if active else 'OFF (holding current stance targets)'}"
        )

    def set_default_pose_mode(self, enabled: bool):
        self._default_pose_mode = enabled
        if enabled:
            self._motion_done = False
            self._hold_targets = None  # re-capture at current stance
        logger.info(f"[MaskedMimicPolicy] default_pose_mode={'ON' if enabled else 'OFF'}")

    def post_step_callback(self, commands=None):
        for cmd in commands or []:
            if cmd in ("[MOTION_RESET]", "[MOTION_FADE_IN]"):
                self.reset()

    # ------------------------------------------------------------------ #
    # Target-time machinery (MaskedMimicControl with a receding horizon)
    # ------------------------------------------------------------------ #
    def _sample_time_step(self):
        last = float(np.max(self._target_times))
        horizon_end = self._t + self._cond_lookahead
        remaining = max(horizon_end - last, 0.0)
        beta = float(self._rng.beta(self._time_alpha, self._time_beta))
        new_t = np.clip(last + beta * remaining, self._t + self._control_dt, horizon_end)
        self._target_times[:-1] = self._target_times[1:]
        self._target_times[-1] = new_t

    def _update_target_times(self):
        if self._target_times is None:
            self._target_times = np.full(self._S, self._t, dtype=np.float64)
            for _ in range(self._S):
                self._sample_time_step()
        if self._t >= self._target_times[0]:
            self._sample_time_step()

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #
    def get_observation(self, env_data, ctrl_data):
        t_start = time.perf_counter()

        # -- clutch: on UNFREEZE (Y toggle-off or 'r' while frozen) the
        #    operator may have moved -- RE-REGISTER so their current pose is
        #    the new zero (fresh frame0 offsets), with the target blend
        #    guaranteeing continuity from the frozen targets. Never resume
        #    with pre-freeze offsets. --
        if self._tracking_active and self._was_paused and not self._paused:
            logger.info(
                "[MaskedMimicPolicy] unfreeze -> re-registering (clutch: "
                "operator's current pose is the new zero)"
            )
            self.reset_alignment()
        self._was_paused = self._paused

        # -- teleop seam: pump the latest world targets. Gated on the tracking
        #    flag (armed/hold ignores teleop entirely) and frozen while paused
        #    (the live-freeze fix pattern). --
        if (
            self._teleop_ref
            and self._tracking_active
            and not self._paused
            and ctrl_data is not None
        ):
            teleop = ctrl_data.get("TeleopCtrl", {})
            world = teleop.get("world_targets", None)
            if world:
                live = {}
                for key, body in WORLD_KEY_TO_BODY.items():
                    if key in world and body in self._cond_bodies:
                        live[body] = np.asarray(world[key][0], dtype=np.float32)
                if all(b in live for b in self._cond_bodies):
                    # XR-recenter guard (B3): a >1 m single-tick jump on any
                    # target mid-tracking is the recenter/boot-shift signature
                    # -- freeze targets (hold the last injected) and warn
                    # loudly until the operator re-registers ('r').
                    if self._live_targets is not None and not self._recenter_frozen:
                        jump = max(
                            float(np.linalg.norm(live[b] - self._live_targets[b]))
                            for b in self._cond_bodies
                        )
                        if jump > 1.0:
                            self._recenter_frozen = True
                            logger.warning(
                                "[MaskedMimicPolicy] TARGET JUMP %.2f m in one tick "
                                "(XR recenter signature) -- FREEZING targets; press "
                                "'r' to re-register before tracking resumes.",
                                jump,
                            )
                    if self._recenter_frozen:
                        now = time.monotonic()
                        if now - self._recenter_warned > 2.0:
                            logger.warning(
                                "[MaskedMimicPolicy] targets frozen after XR-recenter "
                                "jump -- re-register ('r'/X) to resume tracking."
                            )
                            self._recenter_warned = now
                    else:
                        self._live_targets = live
                        root = world.get("root")
                        if root is not None:
                            self._live_root_quat = np.asarray(
                                root[1], dtype=np.float32
                            )

        # -- FK current full-body state --
        base_lin_vel = env_data.get("base_lin_vel", None)
        if base_lin_vel is None:
            base_lin_vel = np.zeros(3, dtype=np.float32)
        cur = self._builder.body_state(
            dof_pos=np.asarray(env_data.dof_pos, dtype=np.float64),
            dof_vel=np.asarray(env_data.dof_vel, dtype=np.float64),
            base_pos=np.asarray(env_data.base_pos, dtype=np.float64),
            base_quat_xyzw=np.asarray(env_data.base_quat, dtype=np.float64),
            base_lin_vel=base_lin_vel,
            base_ang_vel=np.asarray(env_data.base_ang_vel, dtype=np.float64),
        )

        # -- history ring buffer (index 0 = t-dt) --
        if not self._state_hist:
            self._state_hist = [cur] * self._H
        hist = self._state_hist[: self._H]

        # -- clock + target times --
        if not self._paused and not self._default_pose_mode:
            self._update_target_times()
        elif self._target_times is None:
            self._update_target_times()
        time_offsets = (self._target_times - self._t).astype(np.float32)

        # -- conditioning targets --
        targets = self._resolve_targets(cur)

        # ref pos/rot: current state everywhere (finite, masked out), the
        # conditioned bodies' slots overwritten with the injected targets.
        ref_pos = np.tile(cur["pos"][None, None], (1, self._S, 1, 1)).astype(np.float32)
        ref_rot = np.tile(cur["rot"][None, None], (1, self._S, 1, 1)).astype(np.float32)
        for body, pos in targets.items():
            ref_pos[0, :, self._body_names.index(body), :] = pos

        # -- assemble raw context arrays --
        key_to_array = {
            "current.rigid_body_pos": cur["pos"][None],
            "current.rigid_body_rot": cur["rot"][None],
            "noisy.rigid_body_pos": cur["pos"][None],
            "noisy.rigid_body_rot": cur["rot"][None],
            "noisy.rigid_body_vel": cur["vel"][None],
            "noisy.rigid_body_ang_vel": cur["ang_vel"][None],
            "noisy_ground_heights": np.zeros((1,), dtype=np.float32),
            "body_contacts": np.zeros((1, self._num_contact), dtype=np.float32),
            "masked_mimic.ref_pos": ref_pos,
            "masked_mimic.ref_rot": ref_rot,
            "masked_mimic.target_bodies_masks": self._target_bodies_masks,
            "masked_mimic.target_poses_masks": self._target_poses_masks,
            "masked_mimic.time_offsets": time_offsets[None],
            "historical.rigid_body_pos": np.stack([h["pos"] for h in hist])[None],
            "historical.rigid_body_rot": np.stack([h["rot"] for h in hist])[None],
            "historical.rigid_body_vel": np.stack([h["vel"] for h in hist])[None],
            "historical.rigid_body_ang_vel": np.stack([h["ang_vel"] for h in hist])[None],
            "historical.ground_heights": np.zeros((1, self._H), dtype=np.float32),
            "historical.body_contacts": np.zeros((1, self._H, self._num_contact), dtype=np.float32),
            # StateHistoryBuffer semantics: slot 0 of the HISTORICAL view is
            # the action from two policy queries ago (buffer slot 0 holds the
            # action applied this step and is excluded from the view).
            "historical.actions": self._action_buf[1 : self._H + 1][None],
        }

        onnx_inputs = {}
        for name in self._onnx_in_names:
            key = self._onnx_name_to_key.get(name)
            if key and key in key_to_array:
                onnx_inputs[name] = np.ascontiguousarray(
                    key_to_array[key], dtype=np.float32
                )

        ort_out = self._session.run(self._onnx_out_names, onnx_inputs)
        out = dict(zip(self._onnx_out_names, ort_out))
        raw_actions = out["actions"][0].copy()
        pd_targets = out["joint_pos_targets"][0].copy()

        # -- PD target acceleration clamp (same as tracker policy) --
        if (
            self._pd_target_max_accel is not None
            and self._prev_pd is not None
            and self._prev_prev_pd is not None
        ):
            delta = pd_targets - self._prev_pd
            prev_delta = self._prev_pd - self._prev_prev_pd
            accel = np.clip(
                delta - prev_delta, -self._pd_target_max_accel, self._pd_target_max_accel
            )
            pd_targets = self._prev_pd + prev_delta + accel
        self._prev_prev_pd = self._prev_pd
        self._prev_pd = pd_targets.copy()

        alpha = self._action_ema_alpha
        if alpha < 1.0:
            if self._ema_prev is None:
                self._ema_prev = pd_targets.copy()
            pd_targets = alpha * pd_targets + (1.0 - alpha) * self._ema_prev
            self._ema_prev = pd_targets.copy()

        self._stashed_pd_targets = pd_targets.astype(np.float32)

        # -- roll history: this state becomes t-dt, this action a_{t-1} --
        self._state_hist.insert(0, cur)
        del self._state_hist[self._H :]
        self._action_buf[1:] = self._action_buf[:-1]
        self._action_buf[0] = raw_actions
        if not self._paused and not self._default_pose_mode:
            self._t += self._control_dt

        self._latency_ms.append((time.perf_counter() - t_start) * 1e3)
        if len(self._latency_ms) > 3000:
            del self._latency_ms[: len(self._latency_ms) - 3000]

        extras = {"CALLBACK": []}
        return np.zeros(1, dtype=np.float32), extras

    def _resolve_targets(self, cur) -> dict:
        """Return {body: (S,3) world positions} for the conditioned bodies."""
        body_pos = {
            b: cur["pos"][self._body_names.index(b)] for b in self._cond_bodies
        }
        if (
            self._default_pose_mode
            or not self._tracking_active
            or self._live_targets is None
        ):
            # Hold: capture the conditioned bodies' CURRENT world positions
            # once and keep conditioning on them (a fixed stand target).
            if self._hold_targets is None:
                self._hold_targets = {b: p.copy() for b, p in body_pos.items()}
            self._last_injected = {b: p.copy() for b, p in self._hold_targets.items()}
            return {
                b: np.tile(p[None], (self._S, 1)) for b, p in self._hold_targets.items()
            }
        # Live tracking: register on first use (frame0 semantics vs the
        # robot's CURRENT stance). Registration captures per-body anchors AND
        # the operator->robot HEADING yaw delta (B1 fix: the model
        # heading-localizes with the ROBOT's heading only, so any XR-world
        # yaw vs robot world would otherwise survive into relative target
        # directions -- operator "hand forward" must map to ROBOT forward).
        if self._register_offset is None:
            self._register_live0 = {
                b: self._live_targets[b].copy() for b in self._cond_bodies
            }
            self._register_robot0 = {
                b: body_pos[b].copy() for b in self._cond_bodies
            }
            self._register_offset = {
                b: self._register_robot0[b] - self._register_live0[b]
                for b in self._cond_bodies
            }
            robot_yaw = _yaw_of_quat_xyzw(cur["rot"][0])
            if self._live_root_quat is not None:
                op_yaw = _yaw_of_quat_xyzw(self._live_root_quat)
                dyaw = robot_yaw - op_yaw
            else:
                dyaw = 0.0
                logger.warning(
                    "[MaskedMimicPolicy] no operator root orientation on the "
                    "target stream -- registering WITHOUT yaw correction "
                    "(operator-forward may not map to robot-forward)."
                )
            self._register_R = _rot_z(dyaw)
            logger.info(
                "[MaskedMimicPolicy] register yaw: robot %+.1f deg, operator "
                "%+.1f deg -> delta %+.1f deg",
                np.degrees(robot_yaw),
                np.degrees(robot_yaw - dyaw),
                np.degrees(dyaw),
            )
            for b, d in self._register_offset.items():
                logger.info(
                    "[MaskedMimicPolicy] register delta %s: [%+.3f %+.3f %+.3f] m",
                    b, d[0], d[1], d[2],
                )
            # Plausibility guard (B3): per-body frame0 deltas should agree to
            # within body-separation x yaw effects; a big spread means the
            # stream and the robot disagree about the world.
            deltas = list(self._register_offset.values())
            spread = max(
                float(np.linalg.norm(d1 - d2))
                for i, d1 in enumerate(deltas)
                for d2 in deltas[i + 1:]
            ) if len(deltas) > 1 else 0.0
            if spread > 0.5:
                logger.warning(
                    "[MaskedMimicPolicy] REGISTRATION PLAUSIBILITY: per-body "
                    "frame0 deltas disagree by %.2f m (>0.5 m) -- the target "
                    "stream and robot frames look inconsistent; check the "
                    "headset calibration / measured normalization.",
                    spread,
                )
        self._hold_targets = None
        # Injected target = robot anchor + yaw-corrected operator DELTA about
        # the registration origin (identity R reproduces live + offset).
        R = self._register_R if self._register_R is not None else np.eye(3)
        tgt = {
            b: (
                self._register_robot0[b]
                + (R @ (self._live_targets[b] - self._register_live0[b]).astype(np.float64)).astype(np.float32)
            )
            for b in self._cond_bodies
        }
        if self._target_blend_left > 0 and self._target_blend_from is not None:
            # Target-space blend from the last injected targets to the newly
            # registered ones (re-register / clutch resume): alpha 0 -> 1
            # over _target_blend_steps, continuous by construction even if
            # the operator moved while frozen.
            alpha = 1.0 - self._target_blend_left / float(self._target_blend_steps)
            tgt = {
                b: (1.0 - alpha) * self._target_blend_from[b] + alpha * tgt[b]
                for b in self._cond_bodies
            }
            self._target_blend_left -= 1
            if self._target_blend_left == 0:
                self._target_blend_from = None
        self._last_injected = {b: t.copy() for b, t in tgt.items()}
        return {b: np.tile(t[None], (self._S, 1)) for b, t in tgt.items()}

    # ------------------------------------------------------------------ #
    def get_action(self, obs):
        return self._stashed_pd_targets

    def get_init_dof_pos(self):
        if self._init_pose is not None:
            return self._init_pose.copy()
        return self._default_dof_pos.copy()

    def latency_stats(self) -> dict:
        if not self._latency_ms:
            return {}
        arr = np.asarray(self._latency_ms)
        return {
            "n": int(arr.size),
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.percentile(arr, 50)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(arr.max()),
        }
