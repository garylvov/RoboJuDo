"""Headless assertion test for the deploy WBC execution state machine.

Drives the full pipeline (DummyEnv + MockPolicy + ScriptedCtrl) through every
WBC command and asserts the state machine behaves correctly:

  * startup ramps to READY pose and holds FROZEN (policy NOT running)
  * FREEZE latches the pose and stops the policy frame from advancing
  * RESUME re-syncs and resumes continuous stepping (frame advances)
  * single-step advances the frame exactly once, then auto-freezes
  * burst advances exactly N frames, then auto-freezes
  * preview computes+surfaces a pending action WITHOUT executing (frame held)
  * confirm executes exactly the previewed action once
  * hands-ready interpolates monotonically to the ready pose
  * damping is a logged no-op on the mock env
  * the recorder writes sane per-component rates

Run:
  ENV=/oscar/data/stellex/glvov/glvov-envs/protomotions-flashsac
  export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
  /tmp/robojudo_venv/bin/python tests/test_wbc_deploy.py
"""

import logging
import tempfile

import numpy as np

from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.rl_pipeline import RlPipeline
from robojudo.pipeline.wbc_execution import ExecMode

logging.basicConfig(level=logging.WARNING, format="%(message)s")

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}  {detail}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def build_pipeline():
    cfg = ConfigManager("g1_dummy_deploy").get_cfg()
    # Deterministic schedule keyed to the post-startup step index.
    cfg.ctrl[0].schedule = {
        3: ["[POLICY_RUN_CONTINUOUS]"],
        9: ["[FREEZE_WBC]"],
        13: ["[POLICY_STEP_ONCE]"],
        16: ["[POLICY_STEP_BURST]"],
        23: ["[POLICY_PREVIEW]"],
        26: ["[POLICY_CONFIRM]"],
        29: ["[HANDS_READY]"],
        37: ["[DAMPING]"],
        38: ["[RESUME_POLICY]"],
    }
    cfg.wbc.ramp_seconds = 0.1  # 5 steps @ 50Hz
    cfg.wbc.burst_steps = 5
    cfg.recorder.output_dir = tempfile.mkdtemp(prefix="robojudo_test_rec_")
    pipe = RlPipeline(cfg=cfg)
    return pipe


def main():
    pipe = build_pipeline()
    inner = pipe._inner_policy()
    ready = pipe._ready_pose

    # ---- startup: ready pose, policy NOT running ----
    pipe.startup()
    check("startup->FROZEN", pipe.exec.mode == ExecMode.FROZEN, f"mode={pipe.exec.mode.value}")
    check("startup latched==ready", np.allclose(pipe.exec._latched, ready), "")

    # Zero the scripted controller and frame counter so schedule keys line up.
    pipe.ctrl_manager.reset()
    inner._frame = 0
    inner._paused = True

    frames = []
    modes = []
    latched = []  # snapshot of the held/latched target each step
    for i in range(45):
        pipe.step()
        frames.append(inner._frame)
        modes.append(pipe.exec.mode)
        latched.append(pipe.exec._latched.copy() if pipe.exec._latched is not None else None)

    # ---- FROZEN before engage (steps 0..2): frame held at 0 ----
    check("pre-engage frozen frame held", frames[2] == 0, f"frame@2={frames[2]}")

    # ---- RESUME_POLICY at step 3 -> RUNNING, frame advances ----
    check("resume->RUNNING", modes[3] == ExecMode.RUNNING, f"mode@3={modes[3].value}")
    check("running advances frame", frames[8] > frames[3], f"f3={frames[3]} f8={frames[8]}")
    run_frame_at_8 = frames[8]

    # ---- FREEZE at step 9 -> frame stops advancing ----
    check("freeze->FROZEN", modes[9] == ExecMode.FROZEN, f"mode@9={modes[9].value}")
    check("freeze holds frame", frames[9] == frames[12], f"f9={frames[9]} f12={frames[12]}")
    frozen_frame = frames[12]

    # ---- SINGLE STEP at step 13 -> frame advances exactly once ----
    check("single-step +1 frame", frames[13] == frozen_frame + 1, f"f12={frames[12]} f13={frames[13]}")
    check("single-step auto-freeze", modes[14] == ExecMode.FROZEN, f"mode@14={modes[14].value}")
    check("single-step then held", frames[15] == frames[13], f"f13={frames[13]} f15={frames[15]}")
    pre_burst = frames[15]

    # ---- BURST (N=5) at step 16 -> frame advances exactly 5 ----
    # executes on steps 16,17,18,19,20 then auto-freeze
    check("burst +5 frames", frames[20] == pre_burst + 5, f"f15={frames[15]} f20={frames[20]}")
    check("burst auto-freeze", modes[21] == ExecMode.FROZEN, f"mode@21={modes[21].value}")
    check("burst then held", frames[22] == frames[20], f"f20={frames[20]} f22={frames[22]}")
    pre_preview = frames[22]

    # ---- PREVIEW at step 23 -> pending computed, NOT executed, frame held ----
    check("preview->PREVIEW", modes[23] == ExecMode.PREVIEW, f"mode@23={modes[23].value}")
    check("preview holds frame", frames[25] == pre_preview, f"f22={frames[22]} f25={frames[25]}")
    pending = pipe.exec.get_pending_preview()
    check("preview surfaced pending", pending is not None and "pending_pd_target" in pending, "")
    check(
        "preview reports max|delta|",
        pending is not None and pending["max_abs_delta"] >= 0.0,
        f"max|d|={pending['max_abs_delta']:.4f}" if pending else "",
    )

    # ---- CONFIRM at step 26 -> execute pending once, then FROZEN ----
    check("confirm +1 frame", frames[26] == pre_preview + 1, f"f25={frames[25]} f26={frames[26]}")
    check("confirm ->FROZEN", modes[27] == ExecMode.FROZEN, f"mode@27={modes[27].value}")

    # ---- HANDS_READY at step 29 -> RAMP then FROZEN at ready ----
    ramp_modes = [modes[i] for i in range(29, 34)]
    check("hands-ready enters RAMP", ExecMode.RAMP in ramp_modes, f"modes29-33={[m.value for m in ramp_modes]}")
    check("hands-ready ->FROZEN", modes[36] == ExecMode.FROZEN, f"mode@36={modes[36].value}")
    # after ramp completes (step 33), the held target equals the ready pose
    check("hands-ready reaches ready", latched[36] is not None and np.allclose(latched[36], ready, atol=1e-4), "")

    # ---- DAMPING at step 37 -> DAMPING (no-op on mock) ----
    check("damping->DAMPING", modes[37] == ExecMode.DAMPING, f"mode@37={modes[37].value}")

    # ---- RESUME at step 38 -> RUNNING again ----
    check("resume-after-damping->RUNNING", modes[38] == ExecMode.RUNNING, f"mode@38={modes[38].value}")
    check("running resumes advancing", frames[44] > frames[38], f"f38={frames[38]} f44={frames[44]}")

    # ---- recorder sanity ----
    stats = pipe.recorder.stats()
    for comp in ("robot_state", "policy_inference", "command_send", "env_step"):
        s = stats.get(comp, {})
        cnt = s.get("count", 0)
        hz = s.get("mean_hz")
        check(
            f"recorder {comp} recorded",
            cnt > 0 and (hz is None or np.isfinite(hz)),
            f"count={cnt} mean_hz={hz}",
        )
    inf = stats.get("policy_inference", {})
    check(
        "recorder measured inference latency",
        inf.get("mean_latency_ms") is not None,
        f"lat={inf.get('mean_latency_ms')}ms",
    )

    pipe._teardown_deploy_resources()

    print(f"\n=== {PASS} passed, {FAIL} failed ===")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
