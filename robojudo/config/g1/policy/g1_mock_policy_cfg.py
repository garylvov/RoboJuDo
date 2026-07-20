"""Config for the hardware-free MockPolicy (deploy-harness testing)."""

from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig

from ..env.g1_env_cfg import G1_29DoF


class G1MockPolicyCfg(PolicyCfg):
    policy_type: str = "MockPolicy"
    robot: str = "g1"

    disable_autoload: bool = True  # no ONNX/JIT — pure python action
    freq: int = 50

    obs_dof: DoFConfig = G1_29DoF()
    action_dof: DoFConfig = G1_29DoF()

    action_scale: float = 1.0
    action_beta: float = 1.0
