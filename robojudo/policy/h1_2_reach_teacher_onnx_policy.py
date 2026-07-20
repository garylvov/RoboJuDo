"""H1_2 reach-teacher ONNX policy for RoboJuDo sim2sim.

Wraps :class:`~robojudo.policy.onnx_policy.OnnxPolicy` and reconstructs the two
observation groups of
``imprint_isaaclab_ext...h1_2_reach.reach_env_cfg.ReachEnvCfg.ObservationsCfg``
(as trained -- see ``onnx_exports/reach_teacher_frozen/policy.meta.json`` in the
wt_visual_s2r worktree: ``input_names=["policy", "goal"]``, ``obs_group="policy"``,
``obs_dim=73``, ``action_dim=12``, not recurrent) from MuJoCo ``env_data``.

Term-by-term provenance for the 73-dim ``policy`` group (see
``reach_env_cfg.py``'s ``ObservationsCfg.PolicyCfg`` for the IsaacLab-side
definitions, dims confirmed against ``policy.meta.json``'s ``obs_dim=73``):

+-------------------+-----+-------------------------------------------------------------+
| term              | dim | source in RoboJuDo MuJoCo env_data                          |
+-------------------+-----+-------------------------------------------------------------+
| base_lin_vel      |  3  | env_data.base_lin_vel (root frame)                          |
| base_ang_vel      |  3  | env_data.base_ang_vel (root/IMU frame)                      |
| projected_gravity |  3  | quat_rotate_inverse(base_quat, [0,0,-1]) -- computed here   |
| arm_joint_pos     | 14  | env_data.dof_pos[arm subset] - default_pos (14 arm joints)  |
| arm_joint_vel     | 14  | env_data.dof_vel[arm subset]                                |
| wrist_pos_w       |  6  | FK left/right *_wrist_yaw_link pos, root-yaw-local frame    |
| goal_local        |  6  | self._goal_local (configurable static/cycled goal, below)   |
| goal_wrist_error  |  6  | goal_local - actual wrist pos, root-yaw-local frame          |
| wrist_cmd_offset  |  6  | self._cmd_local - goal_local (integrator state, see below)  |
| last_action       | 12  | self.last_action (policy's own previous 12-dim output)      |
+-------------------+-----+-------------------------------------------------------------+
Total: 73.

The 12-dim ``goal`` group (``bimanual_goal_normalized`` in
``reach_env_cfg.py``) is the ACTION that reproduces ``goal_local`` with no
postural change: ``[(left_norm, right_norm) 6, torso/head zeros 6]``, where
``*_norm = (goal_local[...] - box_center) / box_half`` using the SAME
``_LEFT_LOCAL_CENTER`` / ``_RIGHT_LOCAL_CENTER`` / ``_LOCAL_HALF`` affine box
as ``imprint_isaaclab_ext.wbc.action_term`` (duplicated here, NOT imported --
RoboJuDo is a standalone deploy repo and does not depend on
``imprint_isaaclab_ext``; keep these three tuples in sync with that module's
docstring if the training-side goal box ever changes).

GOAL HANDLING: unlike the lift teacher (whose ``object_position`` etc. are
unconditionally zero -- no goal concept applies), the reach teacher's goal is
a first-class, user-settable input. ``H1_2ReachTeacherOnnxPolicyCfg.goal_presets``
holds a small list of reachable local-frame ``[left_xyz, right_xyz]`` poses
(default: box-center "resting reach" pose only). ``[CYCLE_GOAL]`` (bound to an
unused key in ``ctrl_cfgs.py`` -- see ``"z"`` / ``"LB+RB+B"`` / ``"L1+R1+B"``)
advances ``self._goal_idx`` through the list and RE-SEEDS the wrist-command
integrator at the new goal (a deploy-time convenience; training only seeds
once per episode).

KNOWN GAP (action side, mirrors ``H1_2LiftTeacherOnnxPolicy``): the trained
12-dim action (``[left_wrist_xyz, right_wrist_xyz, torso_xyz, head_xyz]``,
affine-mapped local-frame WBC conditioning) is NOT run through the frozen
masked-mimic WBC ONNX (``imprint_isaaclab_ext.wbc.action_term.WbcReachAction``)
that translates it into 27 joint PD targets -- there is no wired frozen-WBC
ONNX in this MuJoCo deploy path. For an end-to-end infra smoke-test, the first
12 action dims are treated as a small delta on top of the 14 arm-joint default
positions (clipped/scaled), same approximation ``H1_2LiftTeacherOnnxPolicy``
uses; this is NOT a faithful reproduction of the trained control law. The
observation-side wrist-command integrator (``self._cmd_local``) IS run
faithfully (``T_t = T_{t-1} + a_t[0:6] * WRIST_DELTA_SCALE``) since it only
costs bookkeeping, not a WBC forward pass.
"""

import logging

import numpy as np

from robojudo.policy import policy_registry
from robojudo.policy.onnx_policy import OnnxPolicy
from robojudo.policy.policy_cfgs import OnnxPolicyCfg

logger = logging.getLogger(__name__)

# Order matches imprint_isaaclab_ext...h1_2_reach.mdp.observations.ARM_JOINT_NAMES
# (joint-type-major, left-then-right) -- the order the ONNX obs vector expects.
_ARM_JOINT_NAMES_ONNX_ORDER = [
    f"{side}_{joint}_joint"
    for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
    for side in ("left", "right")
]

# Duplicated from imprint_isaaclab_ext.wbc.action_term (read-only mirror, see module
# docstring above) -- the local-frame (root pos + yaw) wrist-target affine box.
_FORWARD_CENTER, _FORWARD_HALF = 0.325, 0.225
_LATERAL_HALF = 0.20
_LEFT_LATERAL_CENTER = 0.25
_RIGHT_LATERAL_CENTER = -0.25
_HEIGHT_CENTER, _HEIGHT_HALF = -0.03, 0.53
_LEFT_LOCAL_CENTER = np.array([_FORWARD_CENTER, _LEFT_LATERAL_CENTER, _HEIGHT_CENTER], dtype=np.float32)
_RIGHT_LOCAL_CENTER = np.array([_FORWARD_CENTER, _RIGHT_LATERAL_CENTER, _HEIGHT_CENTER], dtype=np.float32)
_LOCAL_HALF = np.array([_FORWARD_HALF, _LATERAL_HALF, _HEIGHT_HALF], dtype=np.float32)

# T_t = T_{t-1} + a_t * WRIST_DELTA_SCALE (matches action_term.WRIST_DELTA_SCALE).
_WRIST_DELTA_SCALE = 0.02

# Default goal preset: the box CENTER for both wrists -- the "resting reach" pose, guaranteed
# reachable by construction (action==0 reproduces it exactly per bimanual_goal_normalized).
_DEFAULT_GOAL_PRESETS = [
    [
        float(_LEFT_LOCAL_CENTER[0]), float(_LEFT_LOCAL_CENTER[1]), float(_LEFT_LOCAL_CENTER[2]),
        float(_RIGHT_LOCAL_CENTER[0]), float(_RIGHT_LOCAL_CENTER[1]), float(_RIGHT_LOCAL_CENTER[2]),
    ],
    # A second, closer-in / lower preset (still inside the training goal box) to exercise
    # [CYCLE_GOAL] against a visibly different target during the deploy smoke test.
    [
        0.20, 0.15, -0.40,
        0.20, -0.15, -0.40,
    ],
]


def _quat_rotate_inverse_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """World gravity dir [0,0,-1] expressed in the body frame given by quat_xyzw."""
    x, y, z, w = quat_xyzw
    vx, vy, vz = 0.0, 0.0, -1.0
    qvec = np.array([x, y, z])
    t1 = np.array([vx, vy, vz]) * (2.0 * w * w - 1.0)
    cross1 = np.cross(qvec, np.array([vx, vy, vz]))
    t2 = cross1 * (2.0 * w)
    dot = qvec @ np.array([vx, vy, vz])
    t3 = qvec * (2.0 * dot)
    return t1 - t2 + t3


def _yaw_only_local(vec_w: np.ndarray, base_pos: np.ndarray, base_quat_xyzw: np.ndarray) -> np.ndarray:
    """Project vec_w (world pos) into the robot root YAW-only local frame."""
    x, y, z, w = base_quat_xyzw
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    rel = vec_w - base_pos
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rot @ rel


class H1_2ReachTeacherOnnxPolicyCfg(OnnxPolicyCfg):
    policy_type: str = "H1_2ReachTeacherOnnxPolicy"
    robot: str = "h1_2"

    obs_dim: int = 73
    action_dim: int = 12
    is_recurrent: bool = False

    # ONNX has two named inputs ("policy", "goal"); obs_input_name (base class) covers
    # the "policy" group, goal_input_name (added here) covers the "goal" group.
    obs_input_name: str = "policy"
    goal_input_name: str = "goal"

    action_scale: float = 0.25
    action_clip: float | None = 3.0
    action_beta: float = 1.0

    # ==== goal handling (see module docstring) ====
    # Each entry is [left_x, left_y, left_z, right_x, right_y, right_z] in the root's local
    # (position + yaw) frame, meters -- same convention as reach_mdp.bimanual_wrist_pos_w.
    goal_presets: list[list[float]] = _DEFAULT_GOAL_PRESETS
    goal_start_idx: int = 0


@policy_registry.register
class H1_2ReachTeacherOnnxPolicy(OnnxPolicy):
    cfg_policy: H1_2ReachTeacherOnnxPolicyCfg

    def __init__(self, cfg_policy: H1_2ReachTeacherOnnxPolicyCfg, device: str = "cpu"):
        # Set up goal state BEFORE calling super().__init__(): OnnxPolicy.__init__ calls
        # self.reset() internally, and our reset() override reads self._goal_local.
        assert cfg_policy.goal_presets, "goal_presets must be non-empty"
        self._goal_idx = cfg_policy.goal_start_idx % len(cfg_policy.goal_presets)
        self._goal_local = np.asarray(cfg_policy.goal_presets[self._goal_idx], dtype=np.float32)
        self._cmd_local = self._goal_local.copy()
        self._cmd_seeded = False
        self._last_raw_action = np.zeros(cfg_policy.action_dim, dtype=np.float32)
        self._goal_obs_cache = np.zeros(12, dtype=np.float32)

        super().__init__(cfg_policy=cfg_policy, device=device)
        joint_names = list(self.cfg_obs_dof.joint_names)
        self._arm_idx = [joint_names.index(n) for n in _ARM_JOINT_NAMES_ONNX_ORDER]
        self._arm_default = self.default_dof_pos[self._arm_idx]
        logger.info(
            f"[H1_2ReachTeacherOnnxPolicy] initialized: obs_dim={cfg_policy.obs_dim}, "
            f"action_dim={cfg_policy.action_dim}, providers={self.active_providers}, "
            f"goal_idx={self._goal_idx}, goal_local={self._goal_local.tolist()}"
        )

    def reset(self):
        super().reset()
        self._last_raw_action = np.zeros(self.cfg_policy.action_dim, dtype=np.float32)
        self._cmd_local = self._goal_local.copy()
        self._cmd_seeded = False

    def post_step_callback(self, commands: list[str] | None = None):
        super().post_step_callback(commands)
        for cmd in commands or []:
            if cmd == "[CYCLE_GOAL]":
                self._cycle_goal()

    def _cycle_goal(self):
        presets = self.cfg_policy.goal_presets
        self._goal_idx = (self._goal_idx + 1) % len(presets)
        self._goal_local = np.asarray(presets[self._goal_idx], dtype=np.float32)
        # Deploy-time convenience (see module docstring): re-seed the integrator at the new
        # goal so a goal switch doesn't leave the wrist command chasing the OLD target's offset.
        self._cmd_local = self._goal_local.copy()
        logger.info(f"[H1_2ReachTeacherOnnxPolicy] [CYCLE_GOAL] -> idx={self._goal_idx}, goal_local={self._goal_local.tolist()}")

    def get_observation(self, env_data, ctrl_data):
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        dof_vel = np.asarray(env_data.dof_vel, dtype=np.float32)
        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)  # xyzw
        base_pos = np.asarray(env_data.base_pos, dtype=np.float32)

        base_lin_vel = np.asarray(env_data.base_lin_vel, dtype=np.float32)
        base_ang_vel = np.asarray(env_data.base_ang_vel, dtype=np.float32)
        projected_gravity = _quat_rotate_inverse_gravity(base_quat).astype(np.float32)

        arm_pos = dof_pos[self._arm_idx]
        arm_vel = dof_vel[self._arm_idx]
        arm_joint_pos = (arm_pos - self._arm_default).astype(np.float32)
        arm_joint_vel = arm_vel.astype(np.float32)

        fk_info = getattr(env_data, "fk_info", None) or {}
        wrist_actual_w = None
        wrist_pos_w = np.zeros(6, dtype=np.float32)
        if "left_wrist_yaw_link" in fk_info and "right_wrist_yaw_link" in fk_info:
            left_w = np.asarray(fk_info["left_wrist_yaw_link"]["pos"], dtype=np.float32)
            right_w = np.asarray(fk_info["right_wrist_yaw_link"]["pos"], dtype=np.float32)
            left_local = _yaw_only_local(left_w, base_pos, base_quat)
            right_local = _yaw_only_local(right_w, base_pos, base_quat)
            wrist_pos_w = np.concatenate([left_local, right_local]).astype(np.float32)
            wrist_actual_w = wrist_pos_w
        else:
            logger.warning("[H1_2ReachTeacherOnnxPolicy] fk_info missing wrist bodies; wrist_pos_w=0")

        goal_local = self._goal_local.astype(np.float32)
        if wrist_actual_w is not None:
            goal_wrist_error = (goal_local - wrist_actual_w).astype(np.float32)
        else:
            goal_wrist_error = np.zeros(6, dtype=np.float32)

        if not self._cmd_seeded:
            self._cmd_local = goal_local.copy()
            self._cmd_seeded = True
        wrist_cmd_offset = (self._cmd_local - goal_local).astype(np.float32)

        last_action = self._last_raw_action.astype(np.float32)

        obs = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                arm_joint_pos,
                arm_joint_vel,
                wrist_pos_w,
                goal_local,
                goal_wrist_error,
                wrist_cmd_offset,
                last_action,
            ]
        ).astype(np.float32)
        assert obs.shape[0] == self.cfg_policy.obs_dim, f"obs dim mismatch: {obs.shape[0]} != {self.cfg_policy.obs_dim}"

        left_n = (goal_local[0:3] - _LEFT_LOCAL_CENTER) / _LOCAL_HALF
        right_n = (goal_local[3:6] - _RIGHT_LOCAL_CENTER) / _LOCAL_HALF
        goal_obs = np.concatenate([left_n, right_n, np.zeros(6, dtype=np.float32)]).astype(np.float32)
        self._goal_obs_cache = goal_obs

        extras = {"CALLBACK": [], "hand_pose": None}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Override: teacher action (12-dim) is NOT a direct 27-DOF PD target (see gap note)."""
        raw = self._onnx_forward(obs)
        raw = np.clip(raw, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        self._last_raw_action = raw.copy()

        # Faithful obs-side integrator update (cheap bookkeeping, no WBC forward pass needed).
        self._cmd_local = self._cmd_local + raw[0:6] * _WRIST_DELTA_SCALE

        # Action-side approximation (documented gap): dims[0:12] treated as a bounded delta on
        # the first 12 of the 14 arm-joint default positions (wrist_roll/pitch/yaw for one side
        # dropped for the delta, kept at default) so MuJoCo gets a finite, well-formed PD target.
        pd_target = self.default_dof_pos.copy()
        n = min(12, len(self._arm_idx))
        delta = raw[:n] * self.cfg_policy.action_scale
        pd_target[self._arm_idx[:n]] = self._arm_default[:n] + delta
        return pd_target.astype(np.float32)

    def _onnx_forward(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {
            self.obs_input_name: np.expand_dims(obs, axis=0).astype(np.float32),
            self.cfg_policy.goal_input_name: np.expand_dims(self._goal_obs_cache, axis=0).astype(np.float32),
        }
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        out_by_name = dict(zip(self.output_names, ort_outputs))
        actions = np.asarray(out_by_name[self.action_output_name]).squeeze().astype(np.float32)
        return actions

    def get_init_dof_pos(self):
        return self.default_dof_pos.copy()
