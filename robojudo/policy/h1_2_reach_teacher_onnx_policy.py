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

ACTION SIDE (now faithful, two-stage -- mirrors ``H1_2LiftTeacherOnnxPolicy``'s
fix in commit ``65adb57``). The trained 12-dim action
(``[left_wrist_xyz, right_wrist_xyz, torso_xyz, head_xyz]``) is run through the
SAME frozen masked-mimic WBC ONNX (``imprint_isaaclab_ext.wbc``,
``unified_pipeline.onnx``) the teacher was trained against, via
:class:`~imprint_isaaclab_ext.wbc.runner.WbcOnnxRunner`, reproducing
``WbcReachAction.process_actions``'s **INTEGRAL command mode**
(``cfg.use_command=True``, ``cfg.integral=True`` -- the reach task's mode,
distinct from the lift task's BOX mode) and ``_run_wbc_step``'s masked-mimic
conditioning + history buffers bit-for-bit:

  - wrists: the WBC's world-frame target is ``self._cmd_local`` (the
    observation-side integrator this policy already ran faithfully --
    ``T_t = T_{t-1} + a_t[0:6] * WRIST_DELTA_SCALE``, anti-windup clamped to
    ``+-WRIST_CMD_MAX_OFFSET`` (0.8) around ``goal_local``, action_clip=None
    per commit ``749348f``'s parity note), rotated root-yaw-local -> world.
    This reuses the SAME integrator state the obs side already maintains --
    no new bookkeeping, just routing it into the WBC instead of discarding it.
  - torso/head: DELTA from their current world position, root-yaw-local,
    identical mechanism/constants to the lift teacher
    (``_TORSO_DELTA_HALF``/``_HEAD_DELTA_HALF`` = (0.30, 0.30, 0.30)).
  - legs UNMASKED (not conditioned) -- the WBC's own learned prior decides
    stance/balance, exactly as in training and exactly as the lift teacher's
    wiring does.

This replaces the previous "arm-joint-default delta" bypass hack. See
``action_term.py``'s ``elif self.cfg.integral:`` branch (``process_actions``)
for the training-side reference this mirrors.
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
# Frozen masked-mimic WBC (stage 2) -- same import/path-resolution pattern as
# H1_2LiftTeacherOnnxPolicy (robojudo/policy/h1_2_lift_teacher_onnx_policy.py).
# Framework-light (numpy/torch/onnxruntime only), safe to import directly from
# this plain-python RoboJuDo process.
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

# WBC contract body names -- matches action_term.py's LEFT_WRIST_BODY_NAME /
# RIGHT_WRIST_BODY_NAME / TORSO_BODY_NAME / HEAD_BODY_NAME exactly. Ankles are
# deliberately NOT in this list -- legs stay unmasked/unconditioned ("LEGS FLOAT",
# action_term.py lines ~788-800).
_LEFT_WRIST_BODY_NAME = "left_wrist_yaw_link"
_RIGHT_WRIST_BODY_NAME = "right_wrist_yaw_link"
_TORSO_BODY_NAME = "torso_link"
_HEAD_BODY_NAME = "head_aux"

# Torso/head DELTA half-extents -- copied verbatim from action_term.py's module-level
# _TORSO_DELTA_HALF / _HEAD_DELTA_HALF (lines ~151-152); identical for both the reach
# and lift tasks (same WBC contract, same conditionable-body geometry).
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

# Anti-windup bound on the integrator (matches action_term.WRIST_CMD_MAX_OFFSET, action_term.py:187,
# applied at action_term.py:483-486): keeps self._cmd_local within this distance of goal_local so
# the integrator can't diverge/random-walk arbitrarily far from the goal box, while still leaving
# the overshoot channel (up to this bound) fully available -- same rationale as process_actions'
# "no clip on the raw action" (see action_clip note on H1_2ReachTeacherOnnxPolicyCfg above).
_WRIST_CMD_MAX_OFFSET = 0.8

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
    # Training (WbcReachAction.process_actions -- action_term.py:432-453) does NOT clip the raw
    # action; it only sanitizes non-finite values (torch.nan_to_num). Default None here to match;
    # set explicitly to opt into a clip (get_action() still sanitizes NaN/Inf unconditionally
    # either way).
    action_clip: float | None = None
    action_beta: float = 1.0

    # ==== goal handling (see module docstring) ====
    # Each entry is [left_x, left_y, left_z, right_x, right_y, right_z] in the root's local
    # (position + yaw) frame, meters -- same convention as reach_mdp.bimanual_wrist_pos_w.
    goal_presets: list[list[float]] = _DEFAULT_GOAL_PRESETS
    goal_start_idx: int = 0

    # Stage-2 frozen masked-mimic WBC artifact (see module header for path resolution).
    wbc_onnx_path: str = _DEFAULT_WBC_ONNX
    wbc_yaml_path: str = _DEFAULT_WBC_YAML


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

        # ---- Stage 2: the frozen masked-mimic WBC (imprint_isaaclab_ext.wbc) -----------------
        with _wbc_import_lock:
            self.wbc_runner = WbcOnnxRunner(
                onnx_path=cfg_policy.wbc_onnx_path,
                yaml_path=cfg_policy.wbc_yaml_path,
                device="cpu",
            )
        c = self.wbc_runner.contract
        if list(c.joint_names) != joint_names:
            logger.warning(
                "[H1_2ReachTeacherOnnxPolicy] WBC contract joint_names != obs_dof joint_names -- "
                "expected them to be identical (verified at authoring time, same as the lift "
                "teacher); WBC output will be used AS-IS without reindexing. contract=%s obs_dof=%s",
                c.joint_names,
                joint_names,
            )
        self._wbc_body_names = list(c.body_names)  # 29, contract order
        self._wbc_H = c.num_history_steps  # 5
        self._wbc_ground_heights_shape = tuple(c.input_shapes["historical.ground_heights"])  # [1, 5]
        self._wbc_missing_body_warned = False
        self._env_data = None
        self._reset_wbc_history()

        logger.info(
            f"[H1_2ReachTeacherOnnxPolicy] initialized: obs_dim={cfg_policy.obs_dim}, "
            f"action_dim={cfg_policy.action_dim}, providers={self.active_providers}, "
            f"goal_idx={self._goal_idx}, goal_local={self._goal_local.tolist()}, "
            f"wbc_onnx={cfg_policy.wbc_onnx_path}"
        )

    def _reset_wbc_history(self):
        """(Re)allocate the masked-mimic history ring buffers, uninitialized (filled lazily on
        the next ``get_action`` call from the actual current pose -- see the ``_wbc_hist_needs_init``
        check in ``get_action``). Single-env (B=1) equivalent of ``action_term.py``'s per-env
        ``_needs_hist_init`` flag. Verbatim copy of ``H1_2LiftTeacherOnnxPolicy._reset_wbc_history``."""
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
        self._cmd_local = self._goal_local.copy()
        self._cmd_seeded = False
        # Base OnnxPolicy.__init__ calls self.reset() before this subclass's __init__ has built
        # self.wbc_runner -- skip the WBC-history reset on that first (pre-construction) call;
        # __init__ does its own explicit _reset_wbc_history() once the runner exists.
        if hasattr(self, "wbc_runner"):
            self._reset_wbc_history()
        self._env_data = None

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
        # Cached for get_action(), which the pipeline always calls immediately afterwards with
        # the SAME env_data (see robojudo/pipeline/rl_pipeline.py) -- get_action's signature is
        # obs-only (base Policy contract), but stage 2 (the WBC) needs the raw sim state (body
        # FK, root pose) that only env_data carries. Same pattern as H1_2LiftTeacherOnnxPolicy.
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
        """Two-stage control law: task ONNX (12-dim raw) -> WBC conditioning -> frozen
        masked-mimic WBC ONNX -> 27-dim joint PD target.

        Dims [0:6] of the task ONNX's raw output drive the wrist-command INTEGRATOR
        (``self._cmd_local``, mirrors ``action_term.py``'s ``elif self.cfg.integral:`` branch);
        dims [6:9]/[9:12] are torso/head DELTAs from their current world position. All three map
        to world-frame WBC targets, run through the masked-mimic conditioning (legs UNMASKED) and
        the WBC history-buffer bookkeeping, mirroring
        ``imprint_isaaclab_ext/imprint_isaaclab_ext/wbc/action_term.py``'s
        ``WbcReachAction.process_actions``/``_run_wbc_step`` (integral command mode,
        ``use_command=True``, ``integral=True`` -- the mode the reach task trains with) bit-for-bit.
        """
        raw = self._onnx_forward(obs)
        # Mirror action_term.py:507 (`torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)`)
        # -- training only sanitizes non-finite values, it does NOT clip the raw action (see the
        # process_actions docstring on why a clamp would cap the residual's overshoot channel).
        # action_clip defaults to None (see cfg above) so this is a no-op unless explicitly opted into.
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.cfg_policy.action_clip is not None:
            raw = np.clip(raw, -self.cfg_policy.action_clip, self.cfg_policy.action_clip)
        self._last_raw_action = raw.copy()

        # Faithful obs-side integrator update (matches action_term.py:535 exactly).
        self._cmd_local = self._cmd_local + raw[0:6] * _WRIST_DELTA_SCALE
        # Anti-windup (matches action_term.py:541-543, WRIST_CMD_MAX_OFFSET=0.8): clamp the
        # command to stay within _WRIST_CMD_MAX_OFFSET of the goal, so overshoot stays available
        # but the integrator can't diverge or be random-walked into meaningless territory.
        self._cmd_local = self._goal_local + np.clip(
            self._cmd_local - self._goal_local, -_WRIST_CMD_MAX_OFFSET, _WRIST_CMD_MAX_OFFSET
        )

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

        # ---- [0:6] wrist integrator -> world-frame wrist targets (action_term.py:580-593) -----
        # self._cmd_local (root-yaw-local) is the SAME integrator state the obs side already
        # maintains (wrist_cmd_offset = self._cmd_local - goal_local) -- routed here instead of
        # discarded, per the module docstring.
        left_local = self._cmd_local[0:3]
        right_local = self._cmd_local[3:6]
        left_wrist_target_w = base_pos + _yaw_rotate_local_to_world(left_local, base_quat)
        right_wrist_target_w = base_pos + _yaw_rotate_local_to_world(right_local, base_quat)

        # ---- [6:12] torso/head DELTA from current world position (action_term.py:595-612) ------
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
        # mirrors action_term.py:884-888.
        self._wbc_hist_pos = np.concatenate([pos[:, None], self._wbc_hist_pos[:, :-1]], axis=1)
        self._wbc_hist_rot = np.concatenate([rot[:, None], self._wbc_hist_rot[:, :-1]], axis=1)
        self._wbc_hist_vel = np.concatenate([vel[:, None], self._wbc_hist_vel[:, :-1]], axis=1)
        self._wbc_hist_ang_vel = np.concatenate([ang_vel[:, None], self._wbc_hist_ang_vel[:, :-1]], axis=1)
        self._wbc_hist_actions = np.concatenate([wbc_actions[:, None], self._wbc_hist_actions[:, :-1]], axis=1)

        return joint_pos_targets.squeeze(0).astype(np.float32)

    def _gather_wbc_body_state(self, fk_info: dict):
        """Build [1,29,3]/[1,29,4]/[1,29,3]/[1,29,3] (pos, rot-xyzw, lin_vel, ang_vel) arrays in
        WBC contract body order from RoboJuDo's ``fk_info``. Verbatim copy of
        ``H1_2LiftTeacherOnnxPolicy._gather_wbc_body_state`` (identical asset/contract)."""
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
                "[H1_2ReachTeacherOnnxPolicy] fk_info missing WBC contract body(s) %s -- using "
                "zero pose/vel for them (expected to never happen; contract body names were "
                "verified against the MJCF at authoring time).",
                missing,
            )
            self._wbc_missing_body_warned = True
        return pos, rot, vel, ang_vel

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
