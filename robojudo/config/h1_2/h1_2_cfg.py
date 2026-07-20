"""H1_2 pipeline configs for RoboJuDo.

Minimal H1_2 support authored for the RoboJuDo verification prep task
(see /oscar/data/stellex/glvov/imprint/wbc_push/robojudo_prep.md). Mirrors
robojudo/config/g1/g1_cfg.py's ``g1_protomotions_tracker`` entry.
"""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import KeyboardCtrlCfg, ScriptedCtrlCfg, UnitreeCtrlCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.wbc_execution import WbcExecCfg
from robojudo.tools.recorder import RecorderCfg

from .env.h1_2_mujoco_env_cfg import H1_2MujocoEnvCfg
from .env.h1_2_real_env_cfg import H1_2RealEnvCfg, H1_2UnitreeCfg
from .policy.h1_2_lift_teacher_onnx_cfg import H1_2LiftTeacherOnnxCfg
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
_LIFT_TEACHER_DEPLOY_SCHEDULE: dict[int, list[str]] = {
    3: ["[POLICY_RUN_CONTINUOUS]"],  # engage policy (continuous)
    30: ["[FREEZE_WBC]"],  # freeze at current pose
    45: ["[POLICY_STEP_ONCE]"],  # single step then auto-freeze
    60: ["[POLICY_STEP_BURST]"],  # burst N steps then auto-freeze
    90: ["[POLICY_PREVIEW]"],  # compute+surface next action (not executed)
    100: ["[POLICY_CONFIRM]"],  # execute the previewed action once
    115: ["[RESUME_POLICY]"],  # resume continuous stepping
    180: ["[DAMPING]"],  # enter damping (soft)
    190: ["[SHUTDOWN]"],  # clean shutdown (flush recorder)
}


@cfg_registry.register
class h1_2_mujoco_onnx_deploy(RlPipelineCfg):
    """Headless sim2sim of the lift-teacher ONNX deploy flow on the MuJoCo backend.

    GAP (see H1_2LiftTeacherOnnxPolicy module docstring for the full table):
    10/101 obs dims (object_position/object_rel_wrists/object_height) and 24/101
    (finger_joint_pos/vel) are zero-substituted -- this MuJoCo asset has no cube
    body and no Ability-hand fingers. On the action side, the trained 24-dim
    action (12 wbc_reach conditioning + 12 ability_fingers) is NOT run through
    the frozen masked-mimic WBC ONNX the teacher was trained against; this
    config approximates dims[0:12] as a bounded delta on the arm-joint default
    pose purely to exercise the deploy pipeline end-to-end with finite, sane
    actions. This is an infra smoke-test / gap-analysis harness, not a
    faithful sim2sim reproduction of the trained control law.

        python scripts/run_pipeline.py -c h1_2_mujoco_onnx_deploy --max-steps 200
    """

    robot: str = "h1_2"
    env: H1_2MujocoEnvCfg = H1_2MujocoEnvCfg(
        headless=True,
        visualize_extras=False,
        born_place_align=False,
        random_heading=False,
    )
    ctrl: list[ScriptedCtrlCfg] = [
        ScriptedCtrlCfg(schedule=_LIFT_TEACHER_DEPLOY_SCHEDULE),
    ]
    policy: H1_2LiftTeacherOnnxCfg = H1_2LiftTeacherOnnxCfg()
    wbc: WbcExecCfg = WbcExecCfg(
        startup_ready_pose=True,
        ramp_seconds=0.1,
        burst_steps=5,
    )
    recorder: RecorderCfg = RecorderCfg(
        enabled=True,
        output_dir="/tmp/robojudo_rec_lift_teacher",
    )
    run_fullspeed: bool = True
