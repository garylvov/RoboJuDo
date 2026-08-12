import logging
import os
import time

import mujoco
import numpy as np

try:
    import mujoco_viewer
except Exception:  # pragma: no cover - headless deploy hosts have no GL viewer
    mujoco_viewer = None

from robojudo.environment import Environment, env_registry
from robojudo.environment.env_cfgs import MujocoEnvCfg
from robojudo.environment.utils.mujoco_viz import MujocoVisualizer
from robojudo.utils.util_func import quat_rotate_inverse_np, quatToEuler

logger = logging.getLogger(__name__)


@env_registry.register
class MujocoEnv(Environment):
    cfg_env: MujocoEnvCfg

    def __init__(self, cfg_env: MujocoEnvCfg, device="cpu"):
        super().__init__(cfg_env=cfg_env, device=device)

        self.sim_duration = cfg_env.sim_duration
        self.sim_dt = cfg_env.sim_dt
        self.sim_decimation = cfg_env.sim_decimation
        self.control_dt = self.sim_dt * self.sim_decimation

        self.model = mujoco.MjModel.from_xml_path(cfg_env.xml)  # pyright: ignore[reportAttributeAccessIssue]
        self.model.opt.timestep = self.sim_dt

        # Robot free-base joint address (2026-07-20, lift-scene sim2sim task): scene XMLs may
        # now prepend extra free-jointed bodies before the robot `<include>` (e.g. the H1_2 lift
        # task's box -- see assets/robots/h1_2/h1_2_lift_scene.xml's "ORDERING CONTRACT" comment).
        # That keeps `dof_pos`/`dof_vel`'s trailing `[-self.num_dofs:]` slice below correct (the
        # robot's own hinge joints stay the LAST `num_dofs` entries), but it also means the
        # robot's floating-base free joint is no longer guaranteed to be qpos[0:7]/qvel[0:6] --
        # so locate it explicitly as the LAST free joint in the model (the ordering contract's
        # guarantee) instead of hardcoding index 0. Single-robot XMLs with no extra free bodies
        # (G1, the original H1_2 asset, etc.) have exactly one free joint at index 0, so this is
        # byte-identical to the old hardcoded behavior for every existing config.
        free_jids = [
            i
            for i in range(self.model.njnt)
            if self.model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE  # pyright: ignore[reportAttributeAccessIssue]
        ]
        robot_free_jid = free_jids[-1] if free_jids else None
        self._base_qpos_addr = int(self.model.jnt_qposadr[robot_free_jid]) if robot_free_jid is not None else 0
        self._base_qvel_addr = int(self.model.jnt_dofadr[robot_free_jid]) if robot_free_jid is not None else 0

        # Optional scene object (e.g. the lift task's box) -- exposed via base_env.py's
        # object_pos/object_quat/object_lin_vel properties (None when the scene has no body
        # named "object", e.g. every non-lift MuJoCo scene) for obs-side consumers.
        object_body_id = mujoco.mj_name2id(  # pyright: ignore[reportAttributeAccessIssue]
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"  # pyright: ignore[reportAttributeAccessIssue]
        )
        self._object_body_id = object_body_id if object_body_id >= 0 else None
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]

        # Zero passive MJCF joint stiffness/damping (mjModel.jnt_stiffness /
        # dof_damping). Some source MJCFs (e.g. assets/robots/h1_2, copied
        # verbatim from ProtoMotions' own asset) bake in nonzero per-joint
        # <joint stiffness=... damping=...> spring-damper attributes that
        # ProtoMotions' own MujocoSimulator explicitly zeros at load time
        # (protomotions/simulator/mujoco/simulator.py: "_zero_passive_forces
        # -- we manage PD control ourselves, so passive forces would
        # double-count"). RoboJuDo always drives DOFs via its own PD (either
        # explicit torque-per-substep or the implicit position-actuator path
        # below) so leaving passive joint stiffness/damping nonzero silently
        # adds an uncommanded spring pulling every joint toward the MJCF's
        # neutral qpos0, fighting the policy. Found via the H1_2 "falls a
        # few seconds into held-out motion tracking" parity investigation
        # (2026-07-02) -- see robojudo_prep.md. No-op for assets (e.g. G1's)
        # that already have zero passive joint stiffness/damping.
        self.model.jnt_stiffness[:] = 0.0
        self.model.dof_damping[:] = 0.0

        self.use_implicit_pd = cfg_env.use_implicit_pd
        if self.use_implicit_pd:
            self._configure_implicit_pd_actuators(cfg_env)

        # Name-based dof addressing. update() used to read the robot's joint
        # state as qpos[-num_dofs:]/qvel[-num_dofs:], which silently assumes
        # the robot's actuated joints are the LAST joints in the model. That
        # breaks the moment the MJCF is runtime-patched with extra free-joint
        # bodies AFTER the robot (imprint's projectile cubes): the tail slice
        # then returns the cubes' free-joint coordinates as "joint state",
        # feeding garbage observations to the policy from tick 0. Resolve the
        # configured joints' qpos/qvel addresses by NAME once here instead;
        # fall back to the legacy tail slice only if a name is missing.
        self._dof_qpos_idx = None
        self._dof_qvel_idx = None
        joint_names = list(cfg_env.dof.joint_names)
        qpos_idx, qvel_idx = [], []
        for name in joint_names:
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jnt_id < 0:
                logger.warning(
                    f"[MujocoEnv] dof joint '{name}' not found in model -- "
                    "falling back to positional qpos tail slice"
                )
                qpos_idx = None
                break
            qpos_idx.append(int(self.model.jnt_qposadr[jnt_id]))
            qvel_idx.append(int(self.model.jnt_dofadr[jnt_id]))
        if qpos_idx is not None and len(qpos_idx) == self.num_dofs:
            self._dof_qpos_idx = np.asarray(qpos_idx, dtype=np.intp)
            self._dof_qvel_idx = np.asarray(qvel_idx, dtype=np.intp)

        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        # Headless (deploy/sim2sim on a compute node with no X display): skip
        # the interactive on-screen MujocoViewer entirely.  The env still
        # steps physics and serves state exactly the same; only the live GL
        # window is suppressed.  This is the deploy path -- the real robot has
        # no viewer either, so a headless MuJoCo sim2sim exercises the same
        # code path used against hardware.  Optional offscreen frame capture
        # (mujoco.Renderer, EGL/OSMesa) is attempted lazily in save_frame().
        self.headless = getattr(cfg_env, "headless", False) or mujoco_viewer is None
        self._renderer = None
        if self.headless:
            self.viewer = None
            if mujoco_viewer is None:
                logger.warning("[MujocoEnv] mujoco_viewer unavailable -> running HEADLESS (no live window)")
            else:
                logger.warning("[MujocoEnv] headless=True -> running HEADLESS (no live window)")
        else:
            self.viewer = mujoco_viewer.MujocoViewer(
                self.model,
                self.data,
                width=1200,
                height=900,
                hide_menus=True,
            )
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -10.0
            self.viewer.cam.azimuth = 180.0
            # self.viewer._paused = True

        if cfg_env.visualize_extras and self.viewer is not None:
            self.visualizer = MujocoVisualizer(self.viewer)
        else:
            self.visualizer = None

        self.last_time = time.time()
        self.random_heading = cfg_env.random_heading

        self._apply_random_heading()

        self.update()  # get initial state

    def _configure_implicit_pd_actuators(self, cfg_env: MujocoEnvCfg):
        """Convert raw <motor> torque actuators into MuJoCo native position
        (implicit PD) actuators, matching
        protomotions/simulator/mujoco/simulator.py's
        `_configure_actuators_for_pd()` (used when ProtoMotions'
        `MujocoSimulatorConfig.use_implicit_pd=True`, the DEFAULT for
        BUILT_IN_PD checkpoints -- see robojudo_prep.md "epoch_3400 parity
        gap" for how this was discovered: RoboJuDo previously always used
        *explicit* PD -- torque computed in Python each substep and written
        to raw torque actuators -- which is ProtoMotions'
        `use_implicit_pd=False` mode, not what most checkpoints are
        trained/evaluated under).

        force = gainprm[0] * ctrl + biasprm[0] + biasprm[1]*q + biasprm[2]*qd
              = kp * ctrl - kp * q - kd * qd = kp*(ctrl - q) - kd*qd

        After this, `data.ctrl[i]` holds the target POSITION for actuator i
        (not torque) -- MuJoCo computes and clips the PD force internally at
        every physics substep.
        """
        joint_names = cfg_env.dof.joint_names
        stiffness = np.asarray(cfg_env.dof.stiffness, dtype=np.float64)
        damping = np.asarray(cfg_env.dof.damping, dtype=np.float64)
        torque_limits = np.asarray(cfg_env.dof.torque_limits, dtype=np.float64)
        name_to_idx = {name: i for i, name in enumerate(joint_names)}

        for act_idx in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[act_idx, 0]
            jnt_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jnt_name not in name_to_idx:
                logger.warning(f"[MujocoEnv] implicit PD: actuator joint '{jnt_name}' not in dof config, skipping")
                continue
            i = name_to_idx[jnt_name]
            kp = float(stiffness[i])
            kd = float(damping[i])
            effort = float(torque_limits[i])

            self.model.actuator_gainprm[act_idx, 0] = kp
            self.model.actuator_biastype[act_idx] = 1  # mjBIAS_AFFINE
            self.model.actuator_biasprm[act_idx, 0] = 0.0
            self.model.actuator_biasprm[act_idx, 1] = -kp
            self.model.actuator_biasprm[act_idx, 2] = -kd

            self.model.actuator_forcerange[act_idx, 0] = -effort
            self.model.actuator_forcerange[act_idx, 1] = effort
            self.model.actuator_ctrllimited[act_idx] = 0
            self.model.actuator_forcelimited[act_idx] = 1

        logger.info(f"[MujocoEnv] configured {self.model.nu} actuators as implicit-PD position actuators")

    def _apply_random_heading(self):
        """Rotate the root body by a random yaw if random_heading is enabled."""
        if not self.random_heading:
            return
        yaw = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(yaw / 2), np.sin(yaw / 2)
        q = self.data.qpos[3:7].copy()  # MuJoCo [w, x, y, z]
        # Pre-multiply by yaw rotation q_yaw=[c,0,0,s]: q_new = q_yaw ⊗ q
        self.data.qpos[3] = c * q[0] - s * q[3]
        self.data.qpos[4] = c * q[1] - s * q[2]
        self.data.qpos[5] = c * q[2] + s * q[1]
        self.data.qpos[6] = c * q[3] + s * q[0]

    def reborn(self, init_qpos=None):
        if init_qpos is not None:
            self.data.qpos[0:7] = init_qpos
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # pyright: ignore[reportAttributeAccessIssue]
            self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

    def reset(self):
        # Reborn: reset the FULL simulator state (qpos/qvel/ctrl) back to the
        # model's default standing pose so a multi-episode eval starts each
        # episode from a clean stand instead of inheriting the previous
        # episode's (possibly fallen/drifted) end state. Previously reset() only
        # ran the optional born_place_align block and NEVER touched qpos/qvel, so
        # every episode after the first began wherever the last one ended --
        # which silently invalidated multi-episode sim2sim evals (the robot
        # could "start" already collapsed) and masked falls. Found during the
        # 2026-07-20 H1_2 sim2sim standing investigation (see
        # docs/VISUAL_SIM2REAL_DEPLOY.md).
        #
        # The H1_2 MJCFs define no <key> keyframe, so reset to the model's
        # compiled defaults (qpos0 -> pelvis at its <body pos> height, all hinge
        # joints at 0) via mj_resetData, NOT mj_resetDataKeyframe (which requires
        # a keyframe and would raise). This is byte-identical to the freshly
        # constructed init state used at __init__.
        mujoco.mj_resetData(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.data.ctrl[:] = 0.0
        self._apply_random_heading()
        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
        self.update()

        if self.born_place_align:  # TODO: merge
            self.born_place_align = False  # disable during reset
            self.update()
            self.born_place_align = True  # enable after reset
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

    def update(self, simple=False):  # TODO: clean sensors in xml
        """simple: only update dof pos & vel"""
        if self._dof_qpos_idx is not None:
            dof_pos = self.data.qpos[self._dof_qpos_idx].astype(np.float32)
            dof_vel = self.data.qvel[self._dof_qvel_idx].astype(np.float32)
        else:
            dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs :]
            dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs :]

        self._dof_pos = dof_pos.copy()
        self._dof_vel = dof_vel.copy()

        if simple:
            return

        a, va = self._base_qpos_addr, self._base_qvel_addr
        quat = self.data.qpos.astype(np.float32)[a + 3 : a + 7][[1, 2, 3, 0]]
        ang_vel = self.data.qvel.astype(np.float32)[va + 3 : va + 6]
        base_pos = self.data.qpos.astype(np.float32)[a : a + 3]
        lin_vel = self.data.qvel.astype(np.float32)[va : va + 3]

        if self.born_place_align:
            quat, base_pos = self.base_align.align_transform(quat, base_pos)

        lin_vel = quat_rotate_inverse_np(quat, lin_vel)
        rpy = quatToEuler(quat)

        self._base_rpy = rpy.copy()
        self._base_quat = quat.copy()
        self._base_ang_vel = ang_vel.copy()

        self._base_pos = base_pos.copy()
        self._base_lin_vel = lin_vel.copy()

        if self._object_body_id is not None:
            # World-frame pose/velocity of the scene's "object" body (the lift task's box, when
            # the loaded scene xml has one -- see H1_2LiftTeacherOnnxPolicy.get_observation for
            # the yaw-local re-projection consumers actually need). data.xpos/xquat/cvel are
            # forward-kinematics outputs already valid after mj_step (updated by mj_step's own
            # mj_forward pass), so no extra mj_forward call is needed here.
            self._object_pos = self.data.xpos[self._object_body_id].astype(np.float32).copy()
            self._object_quat = self.data.xquat[self._object_body_id].astype(np.float32)[[1, 2, 3, 0]].copy()
            self._object_lin_vel = self.data.cvel[self._object_body_id][3:6].astype(np.float32).copy()

        if self.update_with_fk:
            fk_info = self.fk()
            self._fk_info = fk_info.copy()
            self._torso_ang_vel = fk_info[self._torso_name]["ang_vel"]
            self._torso_quat = fk_info[self._torso_name]["quat"]
            self._torso_pos = fk_info[self._torso_name]["pos"]

    def step(self, pd_target, hand_pose=None):
        assert len(pd_target) == self.num_dofs, "pd_target len should be num_dofs of env"

        if hand_pose is not None and os.environ.get("ROBOJUDO_LOG_HAND_POSE"):
            # Opt-in: this fires EVERY control step (50 Hz), so unconditionally
            # it drowns the rollout -- the operator console and per-episode
            # results scroll past before you can read them.
            # "%s", not a second positional arg: logging treats extra args as
            # printf operands, so the original comma form raised TypeError
            # INSIDE the handler and printed a full traceback per step.
            logger.debug("Hand pose--> %s", hand_pose)

        if self.viewer is not None:
            self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
            if self.viewer.is_alive:
                self.viewer.render()

        if self.use_implicit_pd:
            # Implicit PD: write position targets once; MuJoCo's actuator
            # model (configured in _configure_implicit_pd_actuators)
            # computes and clips PD force internally at every substep --
            # matches ProtoMotions MujocoSimulator's use_implicit_pd=True.
            self.data.ctrl = pd_target
            for _ in range(self.sim_decimation):
                mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
                self.update(simple=True)
        else:
            for _ in range(self.sim_decimation):
                torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                self.data.ctrl = torque

                mujoco.mj_step(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]
                self.update(simple=True)
        self.update(simple=False)

    def save_frame(self, path: str, width: int = 640, height: int = 480) -> bool:
        """Offscreen-render the current sim state to ``path`` (headless).

        Uses ``mujoco.Renderer`` (needs an EGL or OSMesa GL context, selected
        via the ``MUJOCO_GL`` env var).  Returns True on success; on any GL
        failure it logs a warning and returns False so the caller can proceed
        without frames (frames are evidence, never load-bearing for the sim).
        """
        try:
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=height, width=width)
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.distance = 3.0
            cam.elevation = -10.0
            cam.azimuth = 180.0
            cam.lookat = self.data.qpos.astype(np.float64)[:3]
            self._renderer.update_scene(self.data, camera=cam)
            img = self._renderer.render()
            try:
                from PIL import Image

                Image.fromarray(img).save(path)
            except Exception:
                import numpy as _np

                _np.save(path + ".npy", img)
            return True
        except Exception as e:
            logger.warning(f"[MujocoEnv] save_frame failed ({e}); continuing headless without frames")
            return False

    def shutdown(self):
        if self.viewer is not None:
            self.viewer.close()


if __name__ == "__main__":
    from robojudo.config.g1.env.g1_mujuco_env_cfg import G1MujocoEnvCfg

    mujoco_env = MujocoEnv(cfg_env=G1MujocoEnvCfg())
    mujoco_env.viewer._paused = False

    while True:
        # mujoco_env.update()
        mujoco_env.step(np.zeros(mujoco_env.num_dofs))
        time.sleep(0.02)
