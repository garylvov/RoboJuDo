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

from .env.h1_2_mujoco_env_cfg import H1_2MujocoEnvCfg, H1_2MujocoLiftSceneEnvCfg
from .env.h1_2_newton_env_cfg import H1_2NewtonEnvCfg
from .env.h1_2_real_env_cfg import H1_2RealEnvCfg, H1_2UnitreeCfg
from .policy.h1_2_lift_teacher_onnx_cfg import H1_2LiftTeacherOnnxCfg
from .policy.h1_2_protomotions_tracker_cfg import H1_2ProtoMotionsTrackerPolicyCfg
from .policy.h1_2_reach_teacher_onnx_cfg import H1_2ReachTeacherOnnxCfg


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


# Full deploy state-machine walk (startup-ready -> preview -> confirm -> step -> burst ->
# freeze -> resume -> damping -> shutdown) exercised against the reach-teacher ONNX on the
# MuJoCo backend, PLUS two [CYCLE_GOAL] cycles to exercise the goal-switching hook
# (H1_2ReachTeacherOnnxPolicy._cycle_goal / ctrl_cfgs.py's "z" / "LB+RB+B" / "L1+R1+B"
# bindings). Step indices are pipeline steps AFTER startup() (mirrors
# _LIFT_TEACHER_DEPLOY_SCHEDULE above, stretched to a 300-step smoke run).
_REACH_TEACHER_DEPLOY_SCHEDULE: dict[int, list[str]] = {
    3: ["[POLICY_RUN_CONTINUOUS]"],  # engage policy (continuous)
    40: ["[FREEZE_WBC]"],  # freeze at current pose
    55: ["[POLICY_STEP_ONCE]"],  # single step then auto-freeze
    70: ["[POLICY_STEP_BURST]"],  # burst N steps then auto-freeze
    100: ["[POLICY_PREVIEW]"],  # compute+surface next action (not executed)
    110: ["[POLICY_CONFIRM]"],  # execute the previewed action once
    125: ["[RESUME_POLICY]"],  # resume continuous stepping
    150: ["[CYCLE_GOAL]"],  # switch to goal_presets[1] mid-run
    220: ["[CYCLE_GOAL]"],  # switch back to goal_presets[0] (wraps)
    270: ["[DAMPING]"],  # enter damping (soft)
    285: ["[SHUTDOWN]"],  # clean shutdown (flush recorder)
}


@cfg_registry.register
class h1_2_mujoco_reach_deploy(RlPipelineCfg):
    """Headless sim2sim of the reach-teacher ONNX deploy flow on the MuJoCo backend.

    GAP (see H1_2ReachTeacherOnnxPolicy module docstring for the full table): the
    trained 12-dim action ([left_wrist_xyz, right_wrist_xyz, torso_xyz, head_xyz]
    WBC conditioning) is NOT run through the frozen masked-mimic WBC ONNX the teacher
    was trained against; this config approximates dims[0:12] as a bounded delta on the
    arm-joint default pose, same approximation h1_2_mujoco_onnx_deploy uses for the
    lift teacher. This is an infra smoke-test / gap-analysis harness, not a faithful
    sim2sim reproduction of the trained control law.

    GOAL: H1_2ReachTeacherOnnxCfg.goal_presets is a configurable list of reachable
    local-frame wrist-pair poses (default: box-center "resting reach", plus a closer/
    lower second preset). [CYCLE_GOAL] advances through the list -- bound to "z"
    (KeyboardCtrlCfg), "LB+RB+B" (JoystickCtrlCfg), "L1+R1+B" (UnitreeCtrlCfg); this
    config's schedule fires it twice (steps 150, 220) to smoke-test the cycle.

        python scripts/run_pipeline.py -c h1_2_mujoco_reach_deploy --max-steps 300
    """

    robot: str = "h1_2"
    env: H1_2MujocoEnvCfg = H1_2MujocoEnvCfg(
        headless=True,
        visualize_extras=False,
        born_place_align=False,
        random_heading=False,
    )
    ctrl: list[ScriptedCtrlCfg] = [
        ScriptedCtrlCfg(schedule=_REACH_TEACHER_DEPLOY_SCHEDULE),
    ]
    policy: H1_2ReachTeacherOnnxCfg = H1_2ReachTeacherOnnxCfg()
    wbc: WbcExecCfg = WbcExecCfg(
        startup_ready_pose=True,
        ramp_seconds=0.1,
        burst_steps=5,
    )
    recorder: RecorderCfg = RecorderCfg(
        enabled=True,
        output_dir="/tmp/robojudo_rec_reach_teacher",
    )
    run_fullspeed: bool = True


@cfg_registry.register
class h1_2_mujoco_lift_scene_deploy(h1_2_mujoco_onnx_deploy):
    """Same as ``h1_2_mujoco_onnx_deploy``, but the MuJoCo scene is
    ``h1_2_lift_scene.xml`` (robot + table + free-joint box, matching the training scene) via
    ``H1_2MujocoLiftSceneEnvCfg`` -- so ``object_position``/``object_rel_wrists``/
    ``object_height`` (34/101 obs dims) are now REAL instead of zero-substituted. The remaining
    gap (finger_joint_pos/vel -- no Ability-hand finger joints on this MuJoCo asset -- and the
    action-side WBC approximation noted in ``h1_2_mujoco_onnx_deploy``'s docstring) is unchanged.

        python scripts/run_pipeline.py -c h1_2_mujoco_lift_scene_deploy --max-steps 200
    """

    env: H1_2MujocoLiftSceneEnvCfg = H1_2MujocoLiftSceneEnvCfg(
        headless=True,
        visualize_extras=False,
        born_place_align=False,
        random_heading=False,
    )
    recorder: RecorderCfg = RecorderCfg(
        enabled=True,
        output_dir="/tmp/robojudo_rec_lift_teacher_scene",
    )


@cfg_registry.register
class h1_2_newton_deploy(RlPipelineCfg):
    """Headless sim2sim of the lift-teacher ONNX deploy flow on the NEWTON backend.

    Same lift-teacher ONNX policy, WBC state machine, recorder, and scripted deploy
    schedule as ``h1_2_mujoco_onnx_deploy`` -- the ONLY change is the physics backend
    (``H1_2NewtonEnvCfg`` -> Newton's ``SolverMuJoCo`` instead of MuJoCo). This is the
    whole point of the Newton backend: one RoboJuDo deploy harness, swap simulators,
    so a PhysX-trained policy can be evaluated sim2sim in Newton.

    The Newton scene is the composed **H1_2 + Psyonic Ability-hands** MJCF (finger
    DOFs physically present at the wrist_yaw links; see H1_2NewtonEnvCfg.newton_xml).
    Same obs/action gaps as the MuJoCo lift-teacher config apply (see
    H1_2LiftTeacherOnnxPolicy docstring); the finger action head is not yet wired to
    the finger actuators (deploy-remainder).

        CUDA_VISIBLE_DEVICES=3 python scripts/run_pipeline.py -c h1_2_newton_deploy --max-steps 200
    """

    robot: str = "h1_2"
    env: H1_2NewtonEnvCfg = H1_2NewtonEnvCfg(
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
        output_dir="/tmp/robojudo_rec_newton_lift_teacher",
    )
    run_fullspeed: bool = True
