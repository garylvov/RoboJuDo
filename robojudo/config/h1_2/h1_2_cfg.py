"""H1_2 pipeline configs for RoboJuDo.

Minimal H1_2 support authored for the RoboJuDo verification prep task
(see /oscar/data/stellex/glvov/imprint/wbc_push/robojudo_prep.md). Mirrors
robojudo/config/g1/g1_cfg.py's ``g1_protomotions_tracker`` entry.

The six TEACHER deploy configs that used to live here moved to imprint on
2026-08-25 (imprint.robojudo.h1_2_teacher_deploy_cfgs) -- this fork is public
and they were imprint's. What stays is h1_2_protomotions_tracker[_real], which
imprint uses as its DEFAULT_SIM_CONFIG / DEFAULT_REAL_CONFIG.
"""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.h1_2_mujoco_env_cfg import H1_2MujocoEnvCfg
from .env.h1_2_real_env_cfg import H1_2RealEnvCfg, H1_2UnitreeCfg
from .policy.h1_2_protomotions_tracker_cfg import H1_2ProtoMotionsTrackerPolicyCfg




@cfg_registry.register
class h1_2_protomotions_tracker(RlPipelineCfg):
    """ProtoMotions H1_2 tracker with cached motion library.

    Use ``scripts/run_tracker_pipeline.py`` -- it parses ``--onnx-path`` /
    ``--motion-path`` / ``--motion-index``, which the generic
    ``run_pipeline.py`` does not.

    Usage::

        python scripts/run_tracker_pipeline.py -c h1_2_protomotions_tracker \\
            --motion-path assets/motions/h1_2/h1_2_random_subset_tiny.pt \\
            --motion-index 0
    """

    robot: str = "h1_2"
    env: H1_2MujocoEnvCfg = H1_2MujocoEnvCfg(
        born_place_align=False,
        random_heading=False,
    )
    ctrl: list[KeyboardCtrlCfg] = [
        KeyboardCtrlCfg(
            triggers={
                "r": "[MOTION_RESET]",
                "i": "[SIM_REBORN]",
                "o": "[SHUTDOWN]",
                "<": "[MOTION_FADE_IN]",
                ">": "[MOTION_FADE_OUT]",
            },
        ),
    ]

    policy: H1_2ProtoMotionsTrackerPolicyCfg = H1_2ProtoMotionsTrackerPolicyCfg()


@cfg_registry.register
class h1_2_protomotions_tracker_real(h1_2_protomotions_tracker):
    """ProtoMotions tracker on real H1_2 hardware.

    Sim2sim -> sim2real by swapping the env to the real one (same pattern as
    ``g1_protomotions_tracker_real``). ``born_place_align=False`` because the
    policy handles heading alignment itself. Drive it via
    ``imprint.robojudo.gated_inference`` (Enter-to-damp safety gate), not the
    raw ``scripts/run_tracker_pipeline.py``.
    """

    env: H1_2RealEnvCfg = H1_2RealEnvCfg(
        env_type="UnitreeEnv",
        unitree=H1_2UnitreeCfg(
            net_if="enp0s31f6",  # note: change to your network interface
        ),
        born_place_align=False,
    )
    ctrl: list[UnitreeCtrlCfg] = [
        UnitreeCtrlCfg(),
    ]
    do_safety_check: bool = True  # enable safety check for real robot


# Full deploy state-machine walk (startup-ready -> preview -> confirm -> step -> burst ->
# freeze -> resume) exercised against the lift-teacher ONNX on the MuJoCo backend. Step
# indices are pipeline steps AFTER startup() (mirrors g1_deploy_cfg.g1_mujoco_deploy).
