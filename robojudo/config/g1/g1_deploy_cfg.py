"""Deploy-flow demo/harness configs for the G1 (hardware-free).

``g1_dummy_deploy`` wires the DummyEnv + MockPolicy + a ScriptedCtrl that walks
the WBC execution state machine through every command, with the recorder and a
mock Psyonic hand pair enabled.  It runs fully headless (no GPU, no checkpoint,
no X display, no gamepad) so ``scripts/run_pipeline.py`` can exercise the whole
deploy control flow in CI.

    python scripts/run_pipeline.py -c g1_dummy_deploy --max-steps 80
"""

from robojudo.config import cfg_registry
from robojudo.controller.ctrl_cfgs import ScriptedCtrlCfg
from robojudo.environment.psyonic_hand import PsyonicHandCfg
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.wbc_execution import WbcExecCfg
from robojudo.tools.recorder import RecorderCfg

from .env.g1_dummy_env_cfg import G1DummyEnvCfg
from .policy.g1_mock_policy_cfg import G1MockPolicyCfg

# A command schedule that walks the full state machine.  Step indices are
# pipeline steps AFTER startup() (which itself does the ready-pose ramp).
_DEPLOY_DEMO_SCHEDULE: dict[int, list[str]] = {
    3: ["[POLICY_RUN_CONTINUOUS]"],  # engage policy (continuous)
    10: ["[FREEZE_WBC]"],            # freeze at current pose
    16: ["[POLICY_STEP_ONCE]"],      # single step then auto-freeze
    22: ["[POLICY_STEP_BURST]"],     # burst N steps then auto-freeze
    34: ["[POLICY_PREVIEW]"],        # compute+surface next action (not executed)
    40: ["[POLICY_CONFIRM]"],        # execute the previewed action once
    46: ["[HANDS_READY]"],           # ramp to ready pose (hands up)
    58: ["[RESUME_POLICY]"],         # resume continuous stepping
    66: ["[DAMPING]"],               # enter damping (soft)
    72: ["[SHUTDOWN]"],              # clean shutdown (flush recorder + hands)
}


@cfg_registry.register
class g1_dummy_deploy(RlPipelineCfg):
    """Headless deploy control-flow harness config (DummyEnv + MockPolicy)."""

    robot: str = "g1"
    env: G1DummyEnvCfg = G1DummyEnvCfg(
        forward_kinematic=None,
        update_with_fk=False,
        odometry_type="NONE",
        # ready_pose defaults to dof.default_pos when None
    )

    ctrl: list[ScriptedCtrlCfg] = [
        ScriptedCtrlCfg(schedule=_DEPLOY_DEMO_SCHEDULE),
    ]

    policy: G1MockPolicyCfg = G1MockPolicyCfg()

    wbc: WbcExecCfg = WbcExecCfg(
        startup_ready_pose=True,
        ramp_seconds=0.1,  # short ramp for the harness
        burst_steps=5,
    )
    recorder: RecorderCfg = RecorderCfg(
        enabled=True,
        output_dir="/tmp/robojudo_rec",
    )
    psyonic: PsyonicHandCfg = PsyonicHandCfg(
        enabled=True,
        backend="mock",
        left_port="mock-left",
        right_port="mock-right",
    )

    run_fullspeed: bool = True
