"""H1_2 masked-mimic (Track C sparse-teleop student) pipeline configs.

Mirrors ``h1_2_cfg.py``'s tracker entries with the masked-mimic policy.
Drive via ``imprint.robojudo.gated_inference`` with ``--masked-mimic``.
"""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg

from .env.h1_2_mujoco_env_cfg import H1_2MujocoEnvCfg
from .env.h1_2_real_env_cfg import H1_2RealEnvCfg, H1_2UnitreeCfg
from .policy.h1_2_masked_mimic_cfg import H1_2MaskedMimicPolicyCfg


@cfg_registry.register
class h1_2_masked_mimic(RlPipelineCfg):
    """ProtoMotions masked-mimic student, MuJoCo sim."""

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
    policy: H1_2MaskedMimicPolicyCfg = H1_2MaskedMimicPolicyCfg()


@cfg_registry.register
class h1_2_masked_mimic_real(h1_2_masked_mimic):
    """Masked-mimic student on real H1_2 hardware (same env swap as tracker)."""

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
    do_safety_check: bool = True
