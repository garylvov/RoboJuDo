"""Config for the H1_2 reach-teacher ONNX policy (sim2sim, MuJoCo backend).

Points :class:`H1_2ReachTeacherOnnxCfg` at the exported reach-teacher checkpoint
(``obs_dim=73`` "policy" input + ``obs_dim=12`` "goal" input, ``action_dim=12``,
not recurrent -- see
``onnx_exports/reach_teacher_frozen/policy.meta.json`` in the wt_visual_s2r
worktree) and reuses H1_2's real 27-DOF joint config for both ``obs_dof`` and
``action_dof`` (the policy emits a full 27-DOF PD target -- see
``H1_2ReachTeacherOnnxPolicy.get_action`` for the documented action-space gap).
"""

import os

from robojudo.config.h1_2.env.h1_2_env_cfg import H1_2_27DoF
from robojudo.policy.h1_2_reach_teacher_onnx_policy import H1_2ReachTeacherOnnxPolicyCfg

_DEFAULT_ONNX = (
    "/oscar/data/stellex/glvov/wt_visual_s2r/onnx_exports/reach_teacher_frozen/policy.onnx"
)


class H1_2ReachTeacherOnnxCfg(H1_2ReachTeacherOnnxPolicyCfg):
    onnx_path: str = os.environ.get("ROBOJUDO_REACH_TEACHER_ONNX", _DEFAULT_ONNX)

    obs_dof: H1_2_27DoF = H1_2_27DoF()
    action_dof: H1_2_27DoF = H1_2_27DoF()

    # Prefer GPU2 (Isaac/PhysX-dead, plain-CUDA-fine) via ordinary CUDAExecutionProvider;
    # falls back to CPU automatically (OnnxPolicy._resolve_providers / session-create retry)
    # -- see DEPLOY_ONNX.md / trt-tools venv notes (this build's ORT is CPU-only here).
    providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
