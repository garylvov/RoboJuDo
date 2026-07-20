"""NVIDIA Newton (Warp physics) sim2sim backend for RoboJuDo.

Purpose
-------
Evaluate a PhysX-trained policy sim2sim in **Newton** through the *same* RoboJuDo
deploy harness used for the MuJoCo backend (one deploy harness, swap simulators).
:class:`NewtonEnv` mirrors :class:`robojudo.environment.mujoco_env.MujocoEnv`'s
interface exactly (``num_dofs``, DOF order matching the h1_2 asset, explicit or
implicit PD, ``update``/``step``/``reborn``/``reset``/``get_data``), so the WBC
state machine, recorder, and ONNX policy plug in unchanged.

Solver
------
Uses Newton's :class:`newton.solvers.SolverMuJoCo` (the MuJoCo-Warp GPU backend,
``solver="newton"``) -- the canonical Newton loop for an articulated humanoid MJCF
with contacts (see ``newton/examples/robot/example_robot_h1.py`` and
``example_robot_policy.py``). This is *dynamic* stepping (unlike
``src/imprint/sim2sim/newton_visual_eval.py``, which only ``eval_fk``-poses the
robot kinematically for camera rendering).

DOF-order mapping (h1_2_box_feet.xml)
-------------------------------------
Newton's ``add_mjcf(floating=True)`` yields 29 joints:
``floating_base`` (free, 6 qd) + 27 actuated revolute joints + ``head_aux_joint``
(a **fixed / 0-DOF** joint that appears in ``joint_label`` but occupies NO
q/qd coordinate). The 27 actuated coordinates therefore line up 1:1 with
RoboJuDo's ``H1_2_27DoF`` order (left leg, right leg, torso, left arm, right arm).
We still build an explicit name->coordinate map from ``joint_qd_start`` /
``joint_q_start`` so the mapping is correct regardless of the 0-DOF joint's
position in the label list.

PD
--
Implicit PD (default): per-DOF ``joint_target_ke``/``joint_target_kd`` set from
``dof.stiffness``/``damping`` with ``JointTargetMode.POSITION`` on the builder
before ``finalize()`` (the SolverMuJoCo snapshots model arrays at construction);
each step writes position targets to ``control.joint_target_pos`` and MuJoCo-Warp
computes+clips the PD force internally per substep. This is the Newton analogue of
``MujocoEnv``'s ``use_implicit_pd=True`` path (the H1_2 BUILT_IN_PD default).
Newton's MJCF importer leaves ``joint_target_ke/kd`` at 0 (the MJCF's passive
``<joint stiffness=.. damping=..>`` is not re-added as a competing spring), so no
double-counting -- the analogue of ``MujocoEnv`` zeroing ``jnt_stiffness`` /
``dof_damping``.
"""

import logging
import time

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import NewtonEnvCfg
from robojudo.environment.utils.ability_hand_coupling import COUPLED_FINGERS, abh_finger_4bar_q2
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
# Ability-hand finger mapping (composed h1_2_box_feet_ability_hands.xml only).
#
# 6 DRIVEN dof/hand (mirrors the real Ability Hand + h1_2_ability_cfg.py's
# `DRIVEN_FINGER_JOINTS_EXPR`): 4 finger MCP-equivalent joints (this MJCF's
# ``{finger}_q1``) + 2 independently-actuated thumb joints (``thumb_q1``
# rotator, ``thumb_q2`` flexor). This module's own convention (NOT verified
# against the original ProtoMotions training action order -- see module
# docstring in h1_2_lift_teacher_onnx_policy.py) fixes per-hand order as:
#   [index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2]
# ``hand_pose`` passed to :meth:`NewtonEnv.step` is 12-dim = concat(left(6), right(6)).
_HAND_DRIVEN_JOINTS = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q1", "thumb_q2"]
_HAND_SIDES = ("lh_", "rh_")  # left, right (matches compose_h1_2_ability_hands.py prefixes)
# 4 non-thumb fingers: q2 (distal/PIP-equivalent) is NOT independently actuated on real
# hardware -- it is mechanically coupled to q1 by a 4-bar linkage (see ability_hand_coupling.py).
_HAND_COUPLED_JOINTS = [(f"{fin}_q2", f"{fin}_q1") for fin in COUPLED_FINGERS]
# Joint limits (identical across both hands -- verified against both
# ability_hand_{left,right}_large.xml source MJCFs), used to clip commands to
# something the real hand could physically reach.
_HAND_JOINT_LIMITS = {
    "index_q1": (0.0, 1.74), "middle_q1": (0.0, 1.74), "ring_q1": (0.0, 1.74), "pinky_q1": (0.0, 1.74),
    "thumb_q1": (-1.74, 0.0), "thumb_q2": (0.0, 1.74),
}
_HAND_FINGER_STIFFNESS = 8.0  # matches imprint_isaaclab_ext.wbc.h1_2_ability_cfg._FINGER_STIFFNESS
_HAND_FINGER_DAMPING = 0.3  # matches imprint_isaaclab_ext.wbc.h1_2_ability_cfg._FINGER_DAMPING


@env_registry.register
class NewtonEnv(Environment):
    cfg_env: NewtonEnvCfg

    def __init__(self, cfg_env: NewtonEnvCfg, device="cuda:0"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        # Newton deploy is always headless (real-robot code path has no viewer).
        self.headless = True
        self.viewer = None
        self.visualizer = None

        self.use_implicit_pd = cfg_env.use_implicit_pd

        wp.init()
        self.wp_device = wp.get_device(cfg_env.device)
        logger.warning(f"[NewtonEnv] building Newton sim on device={self.wp_device}")

        mjcf = cfg_env.newton_xml or cfg_env.xml

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.add_mjcf(
            mjcf,
            floating=True,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
        )

        # ---- DOF-order mapping: config joint_names -> Newton q/qd coordinates ----
        labels = [lbl.split("/")[-1] for lbl in builder.joint_label]
        # builder.joint_{q,qd}_start have one entry per joint (no trailing total,
        # unlike the finalized model arrays); append the totals as sentinels so
        # per-joint widths are computable for the last joint too. joint_target_ke
        # is length == total qd (6 for the free joint + 1 per actuated DOF).
        q_start = list(builder.joint_q_start) + [len(builder.joint_q)]
        qd_start = list(builder.joint_qd_start) + [len(builder.joint_target_ke)]
        name_to_q = {}
        name_to_qd = {}
        for i, name in enumerate(labels):
            width = qd_start[i + 1] - qd_start[i]
            if width == 1:  # single-DOF actuated joint
                name_to_q[name] = q_start[i]
                name_to_qd[name] = qd_start[i]
        missing = [n for n in self.joint_names if n not in name_to_qd]
        if missing:
            raise RuntimeError(
                f"[NewtonEnv] config joints not found as 1-DOF Newton joints: {missing}\n"
                f"available: {sorted(name_to_qd)}"
            )
        self._cfg_to_q = np.array([name_to_q[n] for n in self.joint_names], dtype=np.int64)
        self._cfg_to_qd = np.array([name_to_qd[n] for n in self.joint_names], dtype=np.int64)
        logger.info(
            f"[NewtonEnv] mapped {self.num_dofs} DOFs; head-aux/0-DOF joints skipped; "
            f"q-idx range [{self._cfg_to_q.min()}..{self._cfg_to_q.max()}]"
        )

        # ---- optional Ability-hand fingers (composed h1_2_box_feet_ability_hands.xml) ----
        self.has_fingers = all(f"{s}{j}" in name_to_qd for s in _HAND_SIDES for j in _HAND_DRIVEN_JOINTS)
        if self.has_fingers:
            # per side: qd indices of the 6 driven joints, in _HAND_DRIVEN_JOINTS order
            self._hand_driven_qd = {s: np.array([name_to_qd[f"{s}{j}"] for j in _HAND_DRIVEN_JOINTS]) for s in _HAND_SIDES}
            self._hand_driven_q = {s: np.array([name_to_q[f"{s}{j}"] for j in _HAND_DRIVEN_JOINTS]) for s in _HAND_SIDES}
            self._hand_driven_limits = np.array([_HAND_JOINT_LIMITS[j] for j in _HAND_DRIVEN_JOINTS])
            # per side: (q2 qd-idx, q1 qd-idx) pairs for the 4 mechanically-coupled fingers
            self._hand_coupled_qd = {
                s: [(name_to_qd[f"{s}{q2}"], name_to_qd[f"{s}{q1}"]) for q2, q1 in _HAND_COUPLED_JOINTS]
                for s in _HAND_SIDES
            }
            n_hand_joints = len(_HAND_DRIVEN_JOINTS) + len(_HAND_COUPLED_JOINTS)
            logger.info(
                f"[NewtonEnv] Ability-hand fingers detected: {n_hand_joints} joints/hand "
                f"({len(_HAND_DRIVEN_JOINTS)} driven + {len(_HAND_COUPLED_JOINTS)} 4-bar-coupled)"
            )
        else:
            self._hand_driven_qd = self._hand_driven_q = self._hand_coupled_qd = None
            logger.info("[NewtonEnv] no Ability-hand finger joints in this MJCF -- finger_joint_pos/vel stay None")

        # ---- implicit PD gains on the builder (snapshotted by the solver) ----
        if self.use_implicit_pd:
            for i in range(self.num_dofs):
                qd = int(self._cfg_to_qd[i])
                builder.joint_target_ke[qd] = float(self.stiffness[i])
                builder.joint_target_kd[qd] = float(self.damping[i])
                builder.joint_target_mode[qd] = int(JointTargetMode.POSITION)
            logger.info(f"[NewtonEnv] configured {self.num_dofs} implicit-PD position targets")
        else:
            # Explicit PD: no position targets; torque written to control.joint_f each substep.
            for i in range(self.num_dofs):
                qd = int(self._cfg_to_qd[i])
                builder.joint_target_mode[qd] = int(JointTargetMode.NONE)
            logger.info(f"[NewtonEnv] explicit-PD mode ({self.num_dofs} DOFs driven via joint_f)")

        # ---- finger PD gains (always implicit-PD position targets, independent of the
        # body's use_implicit_pd choice -- this MJCF has no <equality>/mimic constraint for
        # the coupled distal joints, so BOTH the driven and the 4-bar-coupled joints need an
        # explicit position target every step; see step()'s hand_pose handling) ----
        if self.has_fingers:
            for s in _HAND_SIDES:
                for qd in self._hand_driven_qd[s]:
                    builder.joint_target_ke[int(qd)] = _HAND_FINGER_STIFFNESS
                    builder.joint_target_kd[int(qd)] = _HAND_FINGER_DAMPING
                    builder.joint_target_mode[int(qd)] = int(JointTargetMode.POSITION)
                for q2_qd, _q1_qd in self._hand_coupled_qd[s]:
                    builder.joint_target_ke[int(q2_qd)] = _HAND_FINGER_STIFFNESS
                    builder.joint_target_kd[int(q2_qd)] = _HAND_FINGER_DAMPING
                    builder.joint_target_mode[int(q2_qd)] = int(JointTargetMode.POSITION)
            logger.info(
                f"[NewtonEnv] configured Ability-hand finger PD targets "
                f"(kp={_HAND_FINGER_STIFFNESS}, kd={_HAND_FINGER_DAMPING})"
            )

        self._ground = builder.add_ground_plane()

        self.model = builder.finalize(device=self.wp_device)
        self.model.set_gravity((0.0, 0.0, -9.81))

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            iterations=cfg_env.solver_iterations,
            ls_iterations=cfg_env.solver_ls_iterations,
            nconmax=cfg_env.solver_nconmax,
            njmax=cfg_env.solver_njmax,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        # initial q/qd for reborn
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # reusable host buffer for the full-qd position-target vector. PERSISTENT across
        # step() calls (never reset to 0 wholesale) so a hand_pose commanded on step N
        # holds until a new one is commanded on step N+k, exactly like the real Ability
        # Hand holding its last serial position command (see step()/`_apply_hand_pose`).
        self._target_pos_host = np.zeros(self.control.joint_target_pos.shape[0], dtype=np.float32)
        if self.has_fingers:
            # neutral fully-open initial finger target (q1=0 for all driven joints; the
            # coupled q2's initial target follows the SAME 4-bar law as any other command).
            zero_q2 = float(abh_finger_4bar_q2(0.0))
            for s in _HAND_SIDES:
                for q2_qd, _q1_qd in self._hand_coupled_qd[s]:
                    self._target_pos_host[q2_qd] = zero_q2

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading
        self._apply_random_heading()

        self.update()  # get initial state

    # ------------------------------------------------------------------ #
    def _apply_random_heading(self):
        if not self.random_heading:
            return
        q = self.state_0.joint_q.numpy()
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        # Newton root quat is [x, y, z, w]; pre-multiply by yaw rotation about +Z.
        x, y, z, w = q[3], q[4], q[5], q[6]
        q[3] = c * x - s * y
        q[4] = c * y + s * x
        q[5] = c * z + s * w
        q[6] = c * w - s * z
        self.state_0.joint_q = wp.array(q, dtype=wp.float32, device=self.wp_device)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def reborn(self, init_qpos=None):
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        if init_qpos is not None:
            q = self.state_0.joint_q.numpy()
            q[0:7] = np.asarray(init_qpos, dtype=np.float32)[0:7]
            self.state_0.joint_q = wp.array(q, dtype=wp.float32, device=self.wp_device)
        else:
            self._apply_random_heading()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        wp.copy(self.state_1.joint_q, self.state_0.joint_q)
        wp.copy(self.state_1.joint_qd, self.state_0.joint_qd)
        self.update()

    def reset(self):
        # Reborn: reset the FULL simulator state (joint_q/joint_qd) back to the initial
        # (model-default) pose so a multi-episode eval starts each episode from a clean
        # stand instead of inheriting the previous episode's (possibly fallen/drifted)
        # end state -- mirrors MujocoEnv.reset()'s 2026-07-20 fix (see that method's
        # docstring for the full "previously reset() never touched qpos/qvel" gap). Previously
        # NewtonEnv.reset() only ran the born_place_align dance and never called reborn(),
        # so it had the SAME pre-fix bug MujocoEnv did.
        self.reborn()
        if self.born_place_align:  # TODO: merge (mirrors MujocoEnv)
            self.born_place_align = False
            self.update()
            self.born_place_align = True
            self.set_born_place()
            self.update()

    def set_gains(self, stiffness, damping):
        assert len(stiffness) == self.num_dofs and len(damping) == self.num_dofs
        self.stiffness = np.asarray(stiffness)
        self.damping = np.asarray(damping)

    def self_check(self):
        pass

    def set_born_place(self, quat: np.ndarray | None = None, pos: np.ndarray | None = None):
        quat_ = self.base_quat if quat is None else quat
        pos_ = self.base_pos if pos is None else pos
        super().set_born_place(quat_, pos_)

    # ------------------------------------------------------------------ #
    def update(self, simple=False):
        """simple: only update dof pos & vel (mirrors MujocoEnv.update)."""
        q = self.state_0.joint_q.numpy().astype(np.float32)
        qd = self.state_0.joint_qd.numpy().astype(np.float32)

        self._dof_pos = q[self._cfg_to_q].copy()
        self._dof_vel = qd[self._cfg_to_qd].copy()

        if simple:
            return

        quat = q[3:7].copy()  # Newton root quat is already [x, y, z, w]
        ang_vel = qd[3:6].copy()  # free-joint angular velocity
        base_pos = q[0:3].copy()
        lin_vel = qd[0:3].copy()  # free-joint linear velocity (world frame)

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()
        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self.has_fingers:
            # 12-dim per side [index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2]
            # DRIVEN joint state only (matches ability_fingers action / obs convention --
            # see module header + h1_2_lift_teacher_onnx_policy.py's finger_joint_pos/vel).
            self._finger_joint_pos = np.concatenate(
                [q[self._hand_driven_q[s]] for s in _HAND_SIDES]
            ).astype(np.float32)
            self._finger_joint_vel = np.concatenate(
                [qd[self._hand_driven_qd[s]] for s in _HAND_SIDES]
            ).astype(np.float32)

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    # ------------------------------------------------------------------ #
    def _step_physics(self):
        for _ in range(self.sim_decimation):
            if not self.use_implicit_pd:
                # explicit PD -> torque into control.joint_f
                self.update(simple=True)
                torque = (self._pd_target - self._dof_pos) * self.stiffness - self._dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)
                jf = np.zeros(self.control.joint_f.shape[0], dtype=np.float32)
                jf[self._cfg_to_qd] = torque.astype(np.float32)
                wp.copy(self.control.joint_f, wp.array(jf, dtype=wp.float32, device=self.wp_device))

            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _apply_hand_pose(self, hand_pose):
        """Ability-hand finger command -> position targets (writes ``self._target_pos_host``
        in place; does NOT push to the device -- ``step()`` does that in the same write as
        the body targets). ``hand_pose``: 12-dim ``[left(6), right(6)]``, per-hand order
        ``[index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2]`` (see module header).
        Clips each driven joint to its real range, then derives the 4 non-thumb fingers'
        coupled distal (q2) targets via the exact 4-bar linkage
        (:func:`~robojudo.environment.utils.ability_hand_coupling.abh_finger_4bar_q2`)."""
        if not self.has_fingers:
            logger.debug("[NewtonEnv] hand_pose given but this MJCF has no Ability-hand fingers -- ignored")
            return
        hp = np.asarray(hand_pose, dtype=np.float32).reshape(2, 6)
        for i, s in enumerate(_HAND_SIDES):
            cmd = np.clip(hp[i], self._hand_driven_limits[:, 0], self._hand_driven_limits[:, 1])
            self._target_pos_host[self._hand_driven_qd[s]] = cmd
            # _HAND_DRIVEN_JOINTS[0:4] (the 4 non-thumb q1's) line up 1:1, in order, with
            # _hand_coupled_qd[s] (both built from COUPLED_FINGERS / _HAND_COUPLED_JOINTS).
            q2_targets = abh_finger_4bar_q2(cmd[:4])
            for (q2_qd, _q1_qd), q2v in zip(self._hand_coupled_qd[s], q2_targets):
                self._target_pos_host[q2_qd] = float(q2v)

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"
        pd_target = np.asarray(pd_target, dtype=np.float32)
        self._pd_target = pd_target

        if hand_pose is not None:
            self._apply_hand_pose(hand_pose)

        if self.use_implicit_pd:
            self._target_pos_host[self._cfg_to_qd] = pd_target

        if self.use_implicit_pd or self.has_fingers:
            wp.copy(
                self.control.joint_target_pos,
                wp.array(self._target_pos_host, dtype=wp.float32, device=self.wp_device),
            )

        self._step_physics()
        self.update(simple=False)

    # ------------------------------------------------------------------ #
    def attach_camera(self, vfov_deg: float | None = None):
        """Lazily attach a D435i-equivalent head camera at ``torso_link`` (rides the LIVE,
        physics-stepped sim state -- see ``robojudo/environment/utils/newton_camera.py``).
        Requires newton>=1.3 (raises ImportError with a clear message otherwise -- see that
        module's docstring for the env-gap note). Idempotent: re-attaching just returns the
        existing camera."""
        if getattr(self, "camera", None) is None:
            from robojudo.environment.utils.newton_camera import NewtonHeadCamera

            self.camera = NewtonHeadCamera(self, vfov_deg=vfov_deg)
        return self.camera

    def save_camera_frame(self, path: str, downsample: bool = False):
        """Render (attaching the camera on first use) + save a PNG. ``downsample=True``
        saves the 212x120 (``D435_SIM_RESOLUTION``) frame instead of the native 424x240."""
        cam = self.attach_camera()
        return cam.save_frame(path, downsample=downsample)

    def save_frame(self, path: str, width: int = 640, height: int = 480) -> bool:
        """Newton deploy runs headless with no on-screen viewer. If a head camera has been
        (or can be) attached (see :meth:`attach_camera`), this saves ITS frame as third-person
        "evidence" -- not a real third-person render (Newton's dynamic scene has no separate
        orbit-camera viewer wired here) -- otherwise returns False so callers proceed without
        frames (frames are evidence, never load-bearing for the sim)."""
        try:
            self.save_camera_frame(path)
            return True
        except Exception as e:
            logger.debug(f"[NewtonEnv] save_frame unavailable ({e}); continuing headless without frames")
            return False

    def shutdown(self):
        try:
            wp.synchronize_device(self.wp_device)
        except Exception:
            pass


if __name__ == "__main__":
    from robojudo.config.h1_2.env.h1_2_newton_env_cfg import H1_2NewtonEnvCfg

    env = NewtonEnv(cfg_env=H1_2NewtonEnvCfg())
    for _ in range(50):
        env.step(np.zeros(env.num_dofs))
    print("dof_pos", env.dof_pos[:6], "base_pos", env.base_pos)
