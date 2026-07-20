from robojudo.config import ASSETS_DIR
from robojudo.environment.env_cfgs import MujocoEnvCfg

from .h1_2_env_cfg import H1_2EnvCfg


class H1_2MujocoEnvCfg(H1_2EnvCfg, MujocoEnvCfg):
    env_type: str = MujocoEnvCfg.model_fields["env_type"].default
    is_sim: bool = MujocoEnvCfg.model_fields["is_sim"].default
    # ====== ENV CONFIGURATION ======
    update_with_fk: bool = True
    # ProtoMotions' MujocoSimulatorConfig.use_implicit_pd defaults True for
    # BUILT_IN_PD checkpoints (protomotions/simulator/mujoco/config.py) --
    # match it here for eval-parity with checkpoints trained/evaluated
    # under that default (e.g. h1_2_bm_dr_amass). See robojudo_prep.md.
    use_implicit_pd: bool = True


class H1_2MujocoLiftSceneEnvCfg(H1_2MujocoEnvCfg):
    """Same as ``H1_2MujocoEnvCfg`` but the physics scene is
    ``h1_2_lift_scene.xml`` -- the robot plus a static table + free-joint box matching the
    training scene (see ``imprint_isaaclab_ext...h1_2_lift.lift_env_cfg.LiftSceneCfg``), so
    ``H1_2LiftTeacherOnnxPolicy`` can read real ``object_position``/``object_rel_wrists``/
    ``object_height`` obs instead of the zero-substituted gap (see that policy's module
    docstring). ``forward_kinematic`` is intentionally NOT overridden -- it stays on the
    inherited robot-only ``h1_2_box_feet.xml`` path (``H1_2EnvCfg.forward_kinematic``), because
    ``MujocoKinematics`` assumes the robot's free joint is qpos index 0 / body index 1, which
    only holds for a robot-only asset (the lift scene's box free joint is declared before the
    robot for a different reason -- see the scene xml's "ORDERING CONTRACT" comment -- so it
    must never be the FK model).
    """

    xml: str = (ASSETS_DIR / "robots/h1_2/h1_2_lift_scene.xml").as_posix()
