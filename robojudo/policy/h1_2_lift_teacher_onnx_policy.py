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
| object_position  |  3  | REAL when scene has an "object" body (h1_2_lift_scene.xml), |
|                  |     | else ZERO (bare h1_2_box_feet.xml has no cube/box body)     |
| object_rel_wrists|  6  | REAL alongside object_position (needs both); else ZERO      |
| object_height    |  1  | REAL alongside object_position (raw world z); else ZERO     |
| finger_joint_pos | 12  | ZERO -- no Ability-hand finger joints in h1_2_box_feet.xml  |
| finger_joint_vel | 12  | ZERO -- same                                                |
| last_action      | 24  | self.last_action (policy's own previous 24-dim output)      |
+------------------+-----+------------------------------------------------------------+
Total: 101.

ACTION SIDE (now faithful, two-stage): the trained action space is 24-dim =
12-dim ``wbc_reach`` wrist/torso/head conditioning + 12-dim ``ability_fingers``
direct joint targets. ``ability_fingers`` still has no actuator on this MuJoCo
asset and is discarded (unchanged limitation -- no Ability-hand model here).
The first 12 dims are now run through the SAME frozen masked-mimic WBC ONNX
(``imprint_isaaclab_ext.wbc``, ``unified_pipeline.onnx``) the teacher was
trained against, via :class:`~imprint_isaaclab_ext.wbc.runner.WbcOnnxRunner`,
reproducing ``WbcReachAction.process_actions``'s box-mode wrist/torso/head
mapping and ``_run_wbc_step``'s masked-mimic conditioning + history buffers
bit-for-bit (see ``get_action`` below, and ``action_term.py`` in
``imprint_isaaclab_ext/imprint_isaaclab_ext/wbc/``, which is the reference
this mirrors). Legs are left UNMASKED (not conditioned) -- the WBC's own
learned prior decides stance/balance, exactly as in training. This replaces
the previous "arm-joint-default delta" bypass hack.
"""

import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from robojudo.policy import policy_registry
from robojudo.policy.onnx_policy import OnnxPolicy
from robojudo.policy.policy_cfgs import OnnxPolicyCfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen masked-mimic WBC (stage 2). Framework-light (numpy/torch/onnxruntime
# only, no isaaclab/isaacsim) -- see imprint_isaaclab_ext/imprint_isaaclab_ext/
# wbc/__init__.py's own docstring -- so it's safe to import directly from this
# plain-python RoboJuDo process. Path resolution mirrors
# imprint_isaaclab_ext/imprint_isaaclab_ext/wbc/paths.py (env-var override,
# then a machine-agnostic default export dir).
_WT_VISUAL_S2R_ROOT = Path(
    os.environ.get(
        "IMPRINT_WT_VISUAL_S2R_ROOT",
        "/oscar/data/stellex/glvov/wt_visual_s2r",
    )
)
_IMPRINT_ISAACLAB_EXT_SRC = _WT_VISUAL_S2R_ROOT / "imprint_isaaclab_ext"
if str(_IMPRINT_ISAACLAB_EXT_SRC) not in sys.path and _IMPRINT_ISAACLAB_EXT_SRC.is_dir():
    sys.path.insert(0, str(_IMPRINT_ISAACLAB_EXT_SRC))

from imprint_isaaclab_ext.wbc.runner import WbcOnnxRunner  # noqa: E402

_DEFAULT_WBC_EXPORT_DIR = Path(
    os.environ.get(
        "IMPRINT_WBC_EXPORT_DIR",
        "/oscar/data/stellex/glvov/wbc_data/wbc_checkpoints/mm_trackc_v1_ep330",
    )
)
_DEFAULT_WBC_ONNX = os.environ.get("IMPRINT_WBC_ONNX", str(_DEFAULT_WBC_EXPORT_DIR / "unified_pipeline.onnx"))
_DEFAULT_WBC_YAML = os.environ.get("IMPRINT_WBC_YAML", str(_DEFAULT_WBC_EXPORT_DIR / "unified_pipeline.yaml"))

# WBC contract body names for the two wrists + the two bodies we actively drive as
# "stance" (torso, head_aux) -- matches action_term.py's LEFT_WRIST_BODY_NAME /
# RIGHT_WRIST_BODY_NAME / TORSO_BODY_NAME / HEAD_BODY_NAME exactly. Ankles are
# deliberately NOT in this list -- legs stay unmasked/unconditioned (see module
# docstring and action_term.py's "LEGS FLOAT" comment, lines ~670-682).
_LEFT_WRIST_BODY_NAME = "left_wrist_yaw_link"
_RIGHT_WRIST_BODY_NAME = "right_wrist_yaw_link"
_TORSO_BODY_NAME = "torso_link"
_HEAD_BODY_NAME = "head_aux"

# Box-mode (use_command=False, the lift task's mode -- see
# imprint_isaaclab_ext/imprint_isaaclab_ext/tasks/manager_based/manipulation/h1_2_lift/
# lift_env_cfg.py:426 `WbcReachActionCfg(asset_name="robot", use_command=False)")
# wrist/torso/head geometry constants, copied verbatim from action_term.py's module-level
# constants (lines ~143-253: `_TORSO_DELTA_HALF`, `_HEAD_DELTA_HALF`, `_LEFT_LOCAL_CENTER`,
# `_RIGHT_LOCAL_CENTER`, `_LOCAL_HALF` == `_GOAL_HALF`). These are NOT WbcReachActionCfg
# fields -- they are hardcoded there too, so hardcoding here mirrors the source exactly.
_FORWARD_CENTER, _FORWARD_HALF = 0.325, 0.225
_LATERAL_HALF = 0.20
_LEFT_LATERAL_CENTER = 0.25
_RIGHT_LATERAL_CENTER = -0.25
_HEIGHT_CENTER, _HEIGHT_HALF = -0.03, 0.53
_LEFT_LOCAL_CENTER = np.array([_FORWARD_CENTER, _LEFT_LATERAL_CENTER, _HEIGHT_CENTER], dtype=np.float32)
_RIGHT_LOCAL_CENTER = np.array([_FORWARD_CENTER, _RIGHT_LATERAL_CENTER, _HEIGHT_CENTER], dtype=np.float32)
_LOCAL_HALF = np.array([_FORWARD_HALF, _LATERAL_HALF, _HEIGHT_HALF], dtype=np.float32)
_TORSO_DELTA_HALF = np.array([0.30, 0.30, 0.30], dtype=np.float32)
_HEAD_DELTA_HALF = np.array([0.30, 0.30, 0.30], dtype=np.float32)

_wbc_import_lock = threading.Lock()


def _yaw_rotate_local_to_world(vec_local: np.ndarray, base_quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate a root-yaw-local-frame vector into world frame (rotation only, no translation).

    Inverse of this file's ``_yaw_only_local`` (world -> local, ``-yaw``); this is
    local -> world (``+yaw``). Matches ``action_term.py``'s
    ``quat_apply(yaw_quat(root_quat_w), local)`` for a pure-yaw quaternion.
    """
    x, y, z, w = base_quat_xyzw
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rot @ np.asarray(vec_local, dtype=np.float64)).astype(np.float32)

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
    # Training (WbcReachAction.process_actions, box mode -- action_term.py:432-453) does NOT
    # clip the raw action; it only sanitizes non-finite values (torch.nan_to_num). Default None
    # here to match; set explicitly to opt into a clip (get_action() still sanitizes NaN/Inf
    # unconditionally either way).
    action_clip: float | None = None
    action_beta: float = 1.0

    # Stage-2 frozen masked-mimic WBC artifact (see module header for path resolution).
    wbc_onnx_path: str = _DEFAULT_WBC_ONNX
    wbc_yaml_path: str = _DEFAULT_WBC_YAML


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

        # ---- Stage 2: the frozen masked-mimic WBC (imprint_isaaclab_ext.wbc) -----------------
        with _wbc_import_lock:
            self.wbc_runner = WbcOnnxRunner(
                onnx_path=cfg_policy.wbc_onnx_path,
                yaml_path=cfg_policy.wbc_yaml_path,
                device="cpu",
            )
        c = self.wbc_runner.contract
        # joint_pos_targets is in contract (MJCF) joint order; verified identical (same names,
        # same order) to this policy's own H1_2_27DoF order (h1_2_env_cfg.py), so no remap.
        if list(c.joint_names) != joint_names:
            logger.warning(
                "[H1_2LiftTeacherOnnxPolicy] WBC contract joint_names != obs_dof joint_names -- "
                "expected them to be identical (verified at authoring time); WBC output will be "
                "used AS-IS without reindexing. contract=%s obs_dof=%s",
                c.joint_names,
                joint_names,
            )
        self._wbc_body_names = list(c.body_names)  # 29, contract order
        self._wbc_H = c.num_history_steps  # 5
        self._wbc_ground_heights_shape = tuple(c.input_shapes["historical.ground_heights"])  # [1, 5]
        self._wbc_missing_body_warned = False
        self._reset_wbc_history()

        logger.info(
            f"[H1_2LiftTeacherOnnxPolicy] initialized: obs_dim={cfg_policy.obs_dim}, "
            f"action_dim={cfg_policy.action_dim}, providers={self.active_providers}, "
            f"wbc_onnx={cfg_policy.wbc_onnx_path}"
        )

    def _reset_wbc_history(self):
        """(Re)allocate the masked-mimic history ring buffers, uninitialized (filled lazily on
        the next ``get_action`` call from the actual current pose -- see ``_maybe_init_wbc_history``).
        Single-env (B=1) equivalent of ``action_term.py``'s per-env ``_needs_hist_init`` flag.
        """
        n = len(self._wbc_body_names)
        H = self._wbc_H
        self._wbc_hist_pos = np.zeros((1, H, n, 3), dtype=np.float32)
        self._wbc_hist_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (1, H, n, 1))
        self._wbc_hist_vel = np.zeros((1, H, n, 3), dtype=np.float32)
        self._wbc_hist_ang_vel = np.zeros((1, H, n, 3), dtype=np.float32)
        self._wbc_hist_actions = np.zeros((1, H, self.wbc_runner.contract.num_dofs), dtype=np.float32)
        self._wbc_hist_ground_heights = np.zeros(self._wbc_ground_heights_shape, dtype=np.float32)
        self._wbc_hist_needs_init = True

    def reset(self):
        super().reset()
        self._last_raw_action = np.zeros(self.cfg_policy.action_dim, dtype=np.float32)
        # Base OnnxPolicy.__init__ calls self.reset() before this subclass's __init__ has built
        # self.wbc_runner -- skip the WBC-history reset on that first (pre-construction) call;
        # __init__ does its own explicit _reset_wbc_history() once the runner exists.
        if hasattr(self, "wbc_runner"):
            self._reset_wbc_history()
        self._env_data = None

    def get_observation(self, env_data, ctrl_data):
        # Cached for get_action(), which the pipeline always calls immediately afterwards with
        # the SAME env_data (see robojudo/pipeline/rl_pipeline.py) -- get_action's signature is
        # obs-only (base Policy contract), but stage 2 (the WBC) needs the raw sim state (body
        # FK, root pose) that only env_data carries.
        self._env_data = env_data
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
        left_local = right_local = None
        if "left_wrist_yaw_link" in fk_info and "right_wrist_yaw_link" in fk_info:
            left_w = np.asarray(fk_info["left_wrist_yaw_link"]["pos"], dtype=np.float32)
            right_w = np.asarray(fk_info["right_wrist_yaw_link"]["pos"], dtype=np.float32)
            left_local = _yaw_only_local(left_w, base_pos, base_quat)
            right_local = _yaw_only_local(right_w, base_pos, base_quat)
            wrist_pos_w = np.concatenate([left_local, right_local]).astype(np.float32)
        else:
            logger.warning("[H1_2LiftTeacherOnnxPolicy] fk_info missing wrist bodies; wrist_pos_w=0")

        # object_position/object_rel_wrists/object_height: REAL when the loaded MuJoCo scene has
        # a body named "object" (e.g. assets/robots/h1_2/h1_2_lift_scene.xml -- see
        # MujocoEnv._object_body_id / base_env.py's object_pos property), ZERO-substituted
        # otherwise (the original documented gap, still true for the bare h1_2_box_feet.xml
        # scene, which has no box body). Frame/semantics mirror
        # imprint_isaaclab_ext...h1_2_lift.mdp.observations.object_position_root /
        # object_rel_wrists / object_height exactly: object_position and the wrist terms inside
        # object_rel_wrists are in the SAME root-yaw-local frame as wrist_pos_w above (object_pos
        # world -> yaw-local via the same `_yaw_only_local` helper); object_height is the RAW
        # world z (not yaw-local, not relative).
        object_position = np.zeros(3, dtype=np.float32)
        object_rel_wrists = np.zeros(6, dtype=np.float32)
        object_height = np.zeros(1, dtype=np.float32)
        object_pos_w = getattr(env_data, "object_pos", None)
        if object_pos_w is not None:
            object_pos_w = np.asarray(object_pos_w, dtype=np.float32)
            object_position = _yaw_only_local(object_pos_w, base_pos, base_quat).astype(np.float32)
            object_height = object_pos_w[2:3].astype(np.float32)
            if left_local is not None and right_local is not None:
                object_rel_wrists = np.concatenate(
                    [object_position - left_local, object_position - right_local]
                ).astype(np.float32)
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
        """Two-stage control law: task ONNX (24-dim raw) -> WBC conditioning -> frozen
        masked-mimic WBC ONNX -> 27-dim joint PD target.

        Dims [0:12] of the task ONNX's raw output are the ``wbc_reach`` conditioning; dims
        [12:24] are ``ability_fingers`` (no actuator on this MuJoCo asset -- discarded, same
        documented limitation as before). The [0:12] -> wrist/torso/head world targets mapping,
        the masked-mimic conditioning (legs UNMASKED), and the WBC history-buffer bookkeeping
        all mirror ``imprint_isaaclab_ext/imprint_isaaclab_ext/wbc/action_term.py``'s
        ``WbcReachAction.process_actions``/``_run_wbc_step`` (box mode, ``use_command=False``,
        the mode the lift task trains with) bit-for-bit.
        """
        raw = self._onnx_forward(obs)
        # Mirror action_term.py:450 (`torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)`)
        # -- training only sanitizes non-finite values, it does NOT clip (box mode's raw[0:12] is
        # an unbounded affine coordinate in the goal box; see process_actions' docstring on why a
        # clamp there would silently cap overshoot). action_clip defaults to None (see cfg above)
        # so this is a no-op unless explicitly opted into.
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg_policy.action_clip is not None:
            raw = np.clip(raw, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        self._last_raw_action = raw.copy()

        env_data = self._env_data
        assert env_data is not None, "get_action() called before get_observation() cached env_data"

        base_pos = np.asarray(env_data.base_pos, dtype=np.float32)
        base_quat = np.asarray(env_data.base_quat, dtype=np.float32)  # xyzw

        fk_info = getattr(env_data, "fk_info", None) or {}
        pos, rot, vel, ang_vel = self._gather_wbc_body_state(fk_info)

        if self._wbc_hist_needs_init:
            H = self._wbc_H
            self._wbc_hist_pos[:] = pos[:, None, :, :].repeat(H, axis=1)
            self._wbc_hist_rot[:] = rot[:, None, :, :].repeat(H, axis=1)
            self._wbc_hist_vel[:] = vel[:, None, :, :].repeat(H, axis=1)
            self._wbc_hist_ang_vel[:] = ang_vel[:, None, :, :].repeat(H, axis=1)
            self._wbc_hist_actions[:] = 0.0
            self._wbc_hist_ground_heights[:] = 0.0
            self._wbc_hist_needs_init = False

        # ---- [0:12] wbc_reach conditioning -> world-frame wrist/torso/head targets ----------
        # Box mode (use_command=False): wrist target = local_center + a * local_half, in the
        # root's yaw frame -- action_term.py:464-466. Torso/head: DELTA from current position,
        # root-yaw-local -- action_term.py:526-531.
        left_local = _LEFT_LOCAL_CENTER + raw[0:3] * _LOCAL_HALF
        right_local = _RIGHT_LOCAL_CENTER + raw[3:6] * _LOCAL_HALF
        left_wrist_target_w = base_pos + _yaw_rotate_local_to_world(left_local, base_quat)
        right_wrist_target_w = base_pos + _yaw_rotate_local_to_world(right_local, base_quat)

        torso_idx = self._wbc_body_names.index(_TORSO_BODY_NAME)
        head_idx = self._wbc_body_names.index(_HEAD_BODY_NAME)
        torso_delta = _yaw_rotate_local_to_world(raw[6:9] * _TORSO_DELTA_HALF, base_quat)
        head_delta = _yaw_rotate_local_to_world(raw[9:12] * _HEAD_DELTA_HALF, base_quat)
        torso_target_w = pos[0, torso_idx] + torso_delta
        head_target_w = pos[0, head_idx] + head_delta

        # ---- masked-mimic conditioning: LEGS UNMASKED (only torso/head/wrists conditioned) --
        mm = self.wbc_runner.build_conditioning(
            current_body_pos=pos,
            current_body_rot=rot,
            left_wrist_target_pos=left_wrist_target_w[None, :],
            right_wrist_target_pos=right_wrist_target_w[None, :],
            left_wrist_body_name=_LEFT_WRIST_BODY_NAME,
            right_wrist_body_name=_RIGHT_WRIST_BODY_NAME,
            stance_body_names=[_TORSO_BODY_NAME, _HEAD_BODY_NAME],
            extra_targets={
                _TORSO_BODY_NAME: torso_target_w[None, :],
                _HEAD_BODY_NAME: head_target_w[None, :],
            },
        )

        context = {
            "current.rigid_body_pos": pos,
            "current.rigid_body_rot": rot,
            "current.rigid_body_vel": vel,
            "current.rigid_body_ang_vel": ang_vel,
            "historical.rigid_body_pos": self._wbc_hist_pos,
            "historical.rigid_body_rot": self._wbc_hist_rot,
            "historical.rigid_body_vel": self._wbc_hist_vel,
            "historical.rigid_body_ang_vel": self._wbc_hist_ang_vel,
            "historical.actions": self._wbc_hist_actions,
            "historical.ground_heights": self._wbc_hist_ground_heights,
        }
        context.update(mm)

        out = self.wbc_runner.run(context)
        joint_pos_targets = out["joint_pos_targets"].detach().cpu().numpy().astype(np.float32)
        wbc_actions = out["actions"].detach().cpu().numpy().astype(np.float32)

        # Rotate history: slot 0 == most recent past, slot H-1 == oldest (drop oldest slot) --
        # mirrors action_term.py:718-723.
        self._wbc_hist_pos = np.concatenate([pos[:, None], self._wbc_hist_pos[:, :-1]], axis=1)
        self._wbc_hist_rot = np.concatenate([rot[:, None], self._wbc_hist_rot[:, :-1]], axis=1)
        self._wbc_hist_vel = np.concatenate([vel[:, None], self._wbc_hist_vel[:, :-1]], axis=1)
        self._wbc_hist_ang_vel = np.concatenate([ang_vel[:, None], self._wbc_hist_ang_vel[:, :-1]], axis=1)
        self._wbc_hist_actions = np.concatenate([wbc_actions[:, None], self._wbc_hist_actions[:, :-1]], axis=1)

        return joint_pos_targets.squeeze(0).astype(np.float32)

    def _gather_wbc_body_state(self, fk_info: dict):
        """Build [1,29,3]/[1,29,4]/[1,29,3]/[1,29,3] (pos, rot-xyzw, lin_vel, ang_vel) arrays in
        WBC contract body order from RoboJuDo's ``fk_info`` (``robojudo/tools/kinematics.py``'s
        ``MujocoKinematics.forward()`` -- keyed by body name, already xyzw quat, already carries
        per-body ``lin_vel``/``ang_vel``; body names verified identical to the WBC contract's 29
        contract body names including the virtual ``head_aux`` body)."""
        n = len(self._wbc_body_names)
        pos = np.zeros((1, n, 3), dtype=np.float32)
        rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (1, n, 1))
        vel = np.zeros((1, n, 3), dtype=np.float32)
        ang_vel = np.zeros((1, n, 3), dtype=np.float32)
        missing = []
        for i, name in enumerate(self._wbc_body_names):
            info = fk_info.get(name)
            if info is None:
                missing.append(name)
                continue
            pos[0, i] = info["pos"]
            rot[0, i] = info["quat"]
            vel[0, i] = info["lin_vel"]
            ang_vel[0, i] = info["ang_vel"]
        if missing and not self._wbc_missing_body_warned:
            logger.warning(
                "[H1_2LiftTeacherOnnxPolicy] fk_info missing WBC contract body(s) %s -- using "
                "zero pose/vel for them (expected to never happen; contract body names were "
                "verified against the MJCF at authoring time).",
                missing,
            )
            self._wbc_missing_body_warned = True
        return pos, rot, vel, ang_vel

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
