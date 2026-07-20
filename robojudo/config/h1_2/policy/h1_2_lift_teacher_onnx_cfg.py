"""Config for the H1_2 lift-teacher ONNX policy (sim2sim, MuJoCo backend).

Points :class:`H1_2LiftTeacherOnnxPolicyCfg` at the exported lift-teacher
checkpoint (``obs_dim=101``, ``action_dim=24``, LSTM recurrent, see
``onnx_exports/lift_teacher/lift_teacher_model_4049.meta.json`` in the
wt_visual_s2r worktree) and reuses H1_2's real 27-DOF joint config for both
``obs_dof`` and ``action_dof`` (the policy emits a full 27-DOF PD target --
see ``H1_2LiftTeacherOnnxPolicy.get_action`` for the documented action-space
gap).
"""

import os

from robojudo.config.h1_2.env.h1_2_env_cfg import H1_2_27DoF
from robojudo.policy.h1_2_lift_teacher_onnx_policy import H1_2LiftTeacherOnnxPolicyCfg

_DEFAULT_ONNX = (
    "/oscar/data/stellex/glvov/wt_visual_s2r/onnx_exports/lift_teacher/lift_teacher_model_4049.onnx"
)


class H1_2LiftTeacherOnnxCfg(H1_2LiftTeacherOnnxPolicyCfg):
    onnx_path: str = os.environ.get("ROBOJUDO_LIFT_TEACHER_ONNX", _DEFAULT_ONNX)

    obs_dof: H1_2_27DoF = H1_2_27DoF()
    action_dof: H1_2_27DoF = H1_2_27DoF()

    # Prefer GPU2 (Isaac/PhysX-dead, plain-CUDA-fine) via ordinary CUDAExecutionProvider;
    # falls back to CPU automatically (OnnxPolicy._resolve_providers / session-create retry)
    # since this build's CUDA EP was dropped for a version collision (see trt-tools venv notes).
    providers: list[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
