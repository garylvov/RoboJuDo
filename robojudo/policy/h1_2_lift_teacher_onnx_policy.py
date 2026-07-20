"""H1_2 lift-teacher ONNX policy for RoboJuDo sim2sim.

Wraps :class:`~robojudo.policy.onnx_policy.OnnxPolicy` and reconstructs the
101-dim ``policy`` observation group of
``imprint_isaaclab_ext...h1_2_lift.lift_env_cfg.LiftSprintEnvCfg.ObservationsCfg.PolicyCfg``
(as trained -- see ``obs_group="policy"`` in the exported ``meta.json``) from
MuJoCo ``env_data``.

Term-by-term provenance (see class docstring in ``lift_env_cfg.py``'s
``SprintObservationsCfg.PolicyCfg`` for the IsaacLab-side definitions):

+------------------+-----+------------------------------------------------------------+
| term             | dim | source in RoboJuDo MuJoCo env_data                         |
+------------------+-----+------------------------------------------------------------+
| base_lin_vel     |  3  | env_data.base_lin_vel (root frame)                          |
| base_ang_vel     |  3  | env_data.base_ang_vel (root/IMU frame)                      |
| projected_gravity|  3  | quat_rotate_inverse(base_quat, [0,0,-1]) -- computed here   |
| arm_joint_pos    | 14  | env_data.dof_pos[arm subset] - default_pos (14 arm joints)  |
| arm_joint_vel    | 14  | env_data.dof_vel[arm subset]                                |
| wrist_pos_w      |  6  | FK left/right *_wrist_yaw_link pos, root-yaw-local frame    |
| object_position  |  3  | ZERO -- no cube/box body exists in h1_2_box_feet.xml        |
| object_rel_wrists|  6  | ZERO -- same (no object to compute a wrist-relative vector) |
| object_height    |  1  | ZERO -- same                                                |
| finger_joint_pos | 12  | ZERO -- no Ability-hand finger joints in h1_2_box_feet.xml  |
| finger_joint_vel | 12  | ZERO -- same                                                |
| last_action      | 24  | self.last_action (policy's own previous 24-dim output)      |
+------------------+-----+------------------------------------------------------------+
Total: 101.

KNOWN GAP (action side, not just obs): the trained action space is 24-dim =
12-dim ``wbc_reach`` wrist/torso/head conditioning (consumed by a *separate*
frozen masked-mimic WBC ONNX, which maps it to 27 joint PD targets) + 12-dim
``ability_fingers`` direct joint targets (no actuator on this MuJoCo asset).
This policy does NOT run that second WBC-translation stage -- there is no
Ability-hand model and no wired frozen-WBC ONNX in this MuJoCo deploy path.
For an end-to-end infra smoke-test, the first 12 action dims are treated as a
small delta on top of the 14 arm-joint default positions (clipped/scaled) so
the sim receives a well-formed 27-DOF PD target; this is NOT a faithful
reproduction of the trained control law. See the sprint report for the full
gap writeup.
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


def _quat_rotate_inverse_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """World gravity dir [0,0,-1] expressed in the body frame given by quat_xyzw."""
    x, y, z, w = quat_xyzw
    # Standard quat-rotate-inverse of v=[0,0,-1]: R^T @ v, closed form.
    vx, vy, vz = 0.0, 0.0, -1.0
    qvec = np.array([x, y, z])
    a = vz  # placeholder, replaced below by explicit formula for clarity/perf
    # a = v*(2*w^2 - 1)
    t1 = np.array([vx, vy, vz]) * (2.0 * w * w - 1.0)
    # b = cross(qvec, v) * w * 2
    cross1 = np.cross(qvec, np.array([vx, vy, vz]))
    t2 = cross1 * (2.0 * w)
    # c = qvec * dot(qvec, v) * 2
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


class H1_2LiftTeacherOnnxPolicyCfg(OnnxPolicyCfg):
    policy_type: str = "H1_2LiftTeacherOnnxPolicy"
    robot: str = "h1_2"

    obs_dim: int = 101
    action_dim: int = 24
    is_recurrent: bool = True
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1

    action_scale: float = 0.25
    action_clip: float | None = 3.0
    action_beta: float = 1.0


@policy_registry.register
class H1_2LiftTeacherOnnxPolicy(OnnxPolicy):
    cfg_policy: H1_2LiftTeacherOnnxPolicyCfg

    def __init__(self, cfg_policy: H1_2LiftTeacherOnnxPolicyCfg, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        joint_names = list(self.cfg_obs_dof.joint_names)
        self._arm_idx = [joint_names.index(n) for n in _ARM_JOINT_NAMES_ONNX_ORDER]
        self._arm_default = self.default_dof_pos[self._arm_idx]
        # The ONNX's own 24-dim action (12 wbc_reach + 12 ability_fingers) feeds back into
        # obs as `last_action` -- distinct from self.last_action (27-dim PD target, used by
        # the base Policy class for its own bookkeeping).
        self._last_raw_action = np.zeros(cfg_policy.action_dim, dtype=np.float32)
        logger.info(
            f"[H1_2LiftTeacherOnnxPolicy] initialized: obs_dim={cfg_policy.obs_dim}, "
            f"action_dim={cfg_policy.action_dim}, providers={self.active_providers}"
        )

    def reset(self):
        super().reset()
        self._last_raw_action = np.zeros(self.cfg_policy.action_dim, dtype=np.float32)

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
        wrist_pos_w = np.zeros(6, dtype=np.float32)
        if "left_wrist_yaw_link" in fk_info and "right_wrist_yaw_link" in fk_info:
            left_w = np.asarray(fk_info["left_wrist_yaw_link"]["pos"], dtype=np.float32)
            right_w = np.asarray(fk_info["right_wrist_yaw_link"]["pos"], dtype=np.float32)
            left_local = _yaw_only_local(left_w, base_pos, base_quat)
            right_local = _yaw_only_local(right_w, base_pos, base_quat)
            wrist_pos_w = np.concatenate([left_local, right_local]).astype(np.float32)
        else:
            logger.warning("[H1_2LiftTeacherOnnxPolicy] fk_info missing wrist bodies; wrist_pos_w=0")

        # ZERO-substituted terms (documented gap -- see module docstring table).
        object_position = np.zeros(3, dtype=np.float32)
        object_rel_wrists = np.zeros(6, dtype=np.float32)
        object_height = np.zeros(1, dtype=np.float32)
        finger_joint_pos = np.zeros(12, dtype=np.float32)
        finger_joint_vel = np.zeros(12, dtype=np.float32)

        last_action = self._last_raw_action.astype(np.float32)

        obs = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                arm_joint_pos,
                arm_joint_vel,
                wrist_pos_w,
                object_position,
                object_rel_wrists,
                object_height,
                finger_joint_pos,
                finger_joint_vel,
                last_action,
            ]
        ).astype(np.float32)
        assert obs.shape[0] == self.cfg_policy.obs_dim, f"obs dim mismatch: {obs.shape[0]} != {self.cfg_policy.obs_dim}"

        extras = {"CALLBACK": [], "hand_pose": None}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Override: teacher action (24-dim) is NOT a direct 27-DOF PD target (see gap note).

        GAP (documented, not a faithful reproduction): dims [0:12] (``wbc_reach``
        conditioning) are treated as a small delta on the 14 arm-joint default
        positions (via a fixed linear map onto the first 12 of the 14 arm
        joints -- wrist_roll/pitch/yaw dims dropped for the delta, kept at
        default) purely so the MuJoCo sim receives a finite, bounded PD target
        for the smoke test; dims [12:24] (``ability_fingers``) have no actuator
        on this asset and are discarded entirely. A faithful deploy requires
        wiring the frozen masked-mimic WBC ONNX (imprint_isaaclab_ext.wbc) as a
        second translation stage, which is out of scope here.
        """
        raw = self._onnx_forward(obs)
        raw = np.clip(raw, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        self._last_raw_action = raw.copy()

        pd_target = self.default_dof_pos.copy()
        n = min(12, len(self._arm_idx))
        delta = raw[:n] * self.cfg_policy.action_scale
        pd_target[self._arm_idx[:n]] = self._arm_default[:n] + delta
        return pd_target.astype(np.float32)

    def _onnx_forward(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.obs_input_name: np.expand_dims(obs, axis=0).astype(np.float32)}
        if self.is_recurrent:
            ort_inputs[self.h_in_name] = self._h
            ort_inputs[self.c_in_name] = self._c
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        out_by_name = dict(zip(self.output_names, ort_outputs))
        actions = np.asarray(out_by_name[self.action_output_name]).squeeze().astype(np.float32)
        if self.is_recurrent:
            self._h = np.asarray(out_by_name[self.h_out_name], dtype=np.float32)
            self._c = np.asarray(out_by_name[self.c_out_name], dtype=np.float32)
        return actions

    def get_init_dof_pos(self):
        return self.default_dof_pos.copy()
