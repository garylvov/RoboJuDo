"""Configuration for the H1_2 ProtoMotions tracker policy.

Near-verbatim copy of ``g1_protomotions_tracker_cfg.py`` with
``robot: str = "h1_2"`` so ``ProtoMotionsTrackerPolicyCfg.policy_file``
resolves to ``assets/models/h1_2/protomotions_tracker/<onnx_name>.onnx``.
"""

import os

from robojudo.config import ASSETS_DIR
from robojudo.policy.policy_cfgs import PolicyCfg
from robojudo.tools.tool_cfgs import DoFConfig


class H1_2ProtoMotionsTrackerPolicyCfg(PolicyCfg):
    """Config for :class:`ProtoMotionsTrackerPolicy` on H1_2."""

    policy_type: str = "ProtoMotionsTrackerPolicy"
    robot: str = "h1_2"
    disable_autoload: bool = True

    onnx_name: str = "unified_pipeline"
    onnx_path: str | None = None
    motion_path: str = ""
    motion_index: int = 0

    # imprint teleop (Track 3): when True, ProtoMotionsTrackerPolicy swaps its
    # cached MotionPlayer for a LiveRefSource fed by a TeleopCtrl via ctrl_data.
    # The cached player still loads (for the init pose); this only redirects the
    # reference seam. Set by imprint.robojudo.gated_inference's --teleop mode.
    teleop_ref: bool = False

    @property
    def policy_file(self) -> str:
        if self.onnx_path is not None:
            return self.onnx_path
        # The ONNX is exported fresh from the checkpoint and is no longer a
        # committed binary in the vendored assets tree. Native RoboJuDo runs
        # that don't set ``onnx_path`` can point at the exported file via the
        # ROBOJUDO_H1_2_ONNX env var. The ASSETS_DIR path remains as a
        # last-resort default (only present if a user manually places it there).
        env = os.environ.get("ROBOJUDO_H1_2_ONNX")
        if env:
            return env
        return (ASSETS_DIR / f"models/{self.robot}/protomotions_tracker/{self.onnx_name}.onnx").as_posix()

    action_scale: float = 1.0
    action_clip: float | None = None
    action_beta: float = 1.0

    obs_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
    action_dof: DoFConfig = DoFConfig(joint_names=["placeholder"], default_pos=[0.0])
