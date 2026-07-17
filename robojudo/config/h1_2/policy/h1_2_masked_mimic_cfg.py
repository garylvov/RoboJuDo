"""Configuration for the H1_2 ProtoMotions masked-mimic (Track C) policy.

Mirrors ``h1_2_protomotions_tracker_cfg.py``; the ONNX comes from
``deployment/export_masked_mimic_onnx.py`` (mean-latent VAE student) instead
of the dense-tracker exporter, and the policy consumes sparse teleop world
targets (3point/5point) rather than a retargeted dof reference.
"""

import os

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class H1_2MaskedMimicPolicyCfg(PolicyCfg):
    """Config for :class:`ProtoMotionsMaskedMimicPolicy` on H1_2."""

    policy_type: str = "ProtoMotionsMaskedMimicPolicy"
    robot: str = "h1_2"
    disable_autoload: bool = True

    onnx_name: str = "unified_pipeline"
    onnx_path: str | None = None
    # Init-pose seed only (frame 0 dof_pos); the masked-mimic policy never
    # plays the motion back.
    motion_path: str = ""
    motion_index: int = 0

    # Live teleop: read sparse world targets from ctrl_data["TeleopCtrl"]
    # ["world_targets"] (set by imprint's TeleopProvider). Without it the
    # policy holds its captured stance targets.
    teleop_ref: bool = False
    # Sparse conditioning set: "3point" (head+wrists) or "5point" (+ankles).
    conditioning: str = "3point"
    # Receding look-ahead horizon (s) for the beta-sampled target times.
    target_horizon_s: float = 4.0

    @property
    def policy_file(self) -> str:
        if self.onnx_path is not None:
            return self.onnx_path
        env = os.environ.get("ROBOJUDO_H1_2_MM_ONNX")
        if env:
            return env
        return (
            ASSETS_DIR / f"models/{self.robot}/masked_mimic/{self.onnx_name}.onnx"
        ).as_posix()

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    obs_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
    action_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
