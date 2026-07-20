#!/usr/bin/env python
"""Scripted sim2sim evaluation protocol for RoboJuDo H1_2 deploy configs.

Programmatic wrapper around the SAME wiring ``scripts/run_pipeline.py`` uses
(``ConfigManager`` -> ``RlPipeline``) that drives a registered deploy config
through N repeatable scripted episodes and produces a metrics.json + a
markdown summary table -- the "tested sim2sim" evidence layer referenced in
h1_2_cfg.py's config docstrings.

Per-episode protocol (mirrors the deploy state machine in
``robojudo/pipeline/wbc_execution.py``)::

    startup()                      -- ramp to READY pose, policy FROZEN
    for each episode:
        [POLICY_RUN_CONTINUOUS]    -- engage policy
        run for --episode-seconds
        [FREEZE_WBC]               -- freeze at current pose
        [CYCLE_GOAL]               -- (reach-teacher only, alternate episodes)
        pipeline.reset()           -- fresh episode (env + policy + ScriptedCtrl)

We do NOT reuse the config's own baked-in ``ScriptedCtrlCfg.schedule`` (it
ends in ``[SHUTDOWN]``, a one-shot smoke-test walk through the WBC state
machine) -- we overwrite ``cfg.ctrl[*].schedule`` in-process with our own
episode-relative schedule before constructing the pipeline. This does not
touch ``h1_2_cfg.py`` (owned by other agents); ``ScriptedCtrl._t`` is reset to
0 by ``RlPipeline.reset()`` (via ``ctrl_manager.reset()``), so the same
relative schedule re-fires every episode for free.

Metrics per episode:
  - recorder Hz/latency stats (cumulative snapshot from RoboJuDo's own
    RateRecorder, per component: robot_state / policy_inference /
    command_send / env_step)
  - wall-clock step throughput (independent of the recorder, measured here)
  - raw policy-action-norm stats (mean/p95/max), read from the policy's own
    ``_last_raw_action`` (pre action-scale/WBC network output)
  - DOF excursion from default pose (mean/max over all joints) and wrist
    (FK ``*_wrist_yaw_link``) excursion from the episode's first pose
  - task-specific proxy:
      reach: goal_wrist_error trajectory (mean/max abs, per-axis), sliced out
        of the policy's own 73-dim observation (see
        ``H1_2ReachTeacherOnnxPolicy`` docstring for the term table) -- NOT
        recomputed independently, so it is exactly what the policy saw.
      lift: ``env.object_pos`` if the scene provides one (guarded --
        ``h1_2_box_feet.xml`` has no cube body, so this is expected to be
        unavailable; recorded as such rather than faked).

CAVEAT (see module docstrings in robojudo/policy/h1_2_*_onnx_policy.py):
  - h1_2_mujoco_onnx_deploy (lift teacher): action side runs through the
    frozen masked-mimic WBC ONNX -- a faithful two-stage reproduction of the
    trained control law (arms/wrists), legs unmasked.
  - h1_2_mujoco_reach_deploy (reach teacher): action side ALSO now runs
    through the same frozen masked-mimic WBC ONNX (integral wrist-command
    mode -- see H1_2ReachTeacherOnnxPolicy module docstring) -- a faithful
    two-stage reproduction of the trained reach control law, legs unmasked.
    The previous "arm-joint-default delta" bypass approximation is gone.

Usage::

    python scripts/sim2sim_eval.py -c h1_2_mujoco_onnx_deploy \\
        --episodes 3 --episode-seconds 20 --out-dir /path/to/run --video

    python scripts/sim2sim_eval.py -c h1_2_mujoco_reach_deploy \\
        --episodes 3 --episode-seconds 20 --out-dir /path/to/run --video
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

# MUJOCO_GL / MUJOCO_EGL_DEVICE_ID must be set before mujoco's GL context is
# first created (lazily, inside MujocoEnv.save_frame). Direct EGL device
# addressing (NOT CUDA_VISIBLE_DEVICES masking -- that breaks Vulkan/GL on
# this node) -- see robojudo_prep.md / session notes on gpu3201.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "3")

import numpy as np  # noqa: E402

import robojudo.pipeline  # noqa: E402
from robojudo.config.config_manager import ConfigManager  # noqa: E402
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sim2sim_eval")

# ---------------------------------------------------------------------------
# Read-only obs-slice tables, duplicated from the policy module docstrings
# (H1_2ReachTeacherOnnxPolicy / H1_2LiftTeacherOnnxPolicy) so this script can
# pull task proxies straight out of the observation the policy actually saw,
# without recomputing (and risking drift from) the policy's own math.
# ---------------------------------------------------------------------------
REACH_OBS_DIM = 73
REACH_SLICES = {
    "wrist_pos_w": slice(37, 43),
    "goal_local": slice(43, 49),
    "goal_wrist_error": slice(49, 55),
    "wrist_cmd_offset": slice(55, 61),
}
LIFT_OBS_DIM = 101
LIFT_SLICES = {
    "wrist_pos_w": slice(37, 43),
    "object_position": slice(43, 46),
    "object_rel_wrists": slice(46, 52),
    "object_height": slice(52, 53),
}


def build_pipeline(config_name: str, episode_steps: int, freeze_margin_steps: int = 10):
    """ConfigManager + pipeline construction, matching scripts/run_pipeline.py's
    wiring, with the ScriptedCtrl schedule swapped for our episode protocol."""
    config_manager = ConfigManager(config_name=config_name)
    cfg: RlPipelineCfg = config_manager.get_cfg()

    resume_step = 3
    freeze_step = max(resume_step + 1, episode_steps - freeze_margin_steps)
    schedule = {resume_step: ["[POLICY_RUN_CONTINUOUS]"], freeze_step: ["[FREEZE_WBC]"]}
    for ctrl_cfg in cfg.ctrl:
        if hasattr(ctrl_cfg, "schedule"):
            ctrl_cfg.schedule = dict(schedule)

    pipeline_type = cfg.pipeline_type
    pipeline_class = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Building pipeline: {pipeline_type} for config={config_name}, "
                f"episode_steps={episode_steps}, schedule={schedule}")
    pipeline = pipeline_class(cfg=cfg)
    return pipeline, cfg


def _pstats(arr: list[float]) -> dict:
    if not arr:
        return {"mean": None, "p95": None, "max": None}
    a = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
    }


def run_episode(pipeline, cfg, episode_idx: int, episode_steps: int, is_reach: bool,
                 video_writer=None, video_every_n: int = 2):
    """Drive `episode_steps` pipeline.step() calls, capturing per-step metrics."""
    inner_policy = pipeline._inner_policy()

    # Wrap PolicyWrapper.get_observation once per episode to stash the exact
    # obs vector the policy consumed this step (adapted-DOF, final obs) --
    # runtime instrumentation on OUR pipeline instance, not an edit to any
    # shared robojudo file.
    last_obs_holder = {"obs": None}
    orig_get_observation = pipeline.policy.get_observation

    def _capture_get_observation(env_data, ctrl_data):
        obs, extras = orig_get_observation(env_data, ctrl_data)
        last_obs_holder["obs"] = obs
        return obs, extras

    pipeline.policy.get_observation = _capture_get_observation

    raw_action_norms = []
    dof_excursion = []
    wrist_excursion = []
    goal_wrist_error_abs = []  # reach only
    object_pos_samples = []  # lift only, if available

    default_pos = np.asarray(pipeline.env.dof_cfg.default_pos, dtype=np.float32)
    first_wrist_w = None
    object_pos_available = False

    wall_start = time.time()
    for t in range(episode_steps):
        pipeline.step()

        raw_action = getattr(inner_policy, "_last_raw_action", None)
        if raw_action is not None:
            raw_action_norms.append(float(np.linalg.norm(np.asarray(raw_action))))

        dof_pos = pipeline.env.dof_pos
        dof_excursion.append(float(np.max(np.abs(np.asarray(dof_pos) - default_pos))))

        fk_info = pipeline.env.fk_info or {}
        if "left_wrist_yaw_link" in fk_info and "right_wrist_yaw_link" in fk_info:
            wrist_w = np.concatenate(
                [
                    np.asarray(fk_info["left_wrist_yaw_link"]["pos"]),
                    np.asarray(fk_info["right_wrist_yaw_link"]["pos"]),
                ]
            )
            if first_wrist_w is None:
                first_wrist_w = wrist_w.copy()
            wrist_excursion.append(float(np.max(np.abs(wrist_w - first_wrist_w))))

        obs = last_obs_holder["obs"]
        if is_reach and obs is not None and obs.shape[0] == REACH_OBS_DIM:
            err = obs[REACH_SLICES["goal_wrist_error"]]
            goal_wrist_error_abs.append(np.abs(err).tolist())

        object_pos = getattr(pipeline.env, "object_pos", None)
        if object_pos is not None:
            object_pos_available = True
            object_pos_samples.append(np.asarray(object_pos).tolist())

        if video_writer is not None and hasattr(pipeline.env, "save_frame") and t % video_every_n == 0:
            frame_path = video_writer["tmp_dir"] / f"frame_{t:05d}.png"
            ok = pipeline.env.save_frame(str(frame_path), width=video_writer["width"], height=video_writer["height"])
            if ok:
                video_writer["frames"].append(frame_path)
            else:
                video_writer["render_failed"] = True

    wall_elapsed = time.time() - wall_start

    # Post-episode WBC transitions: FREEZE fires from the schedule mid-loop
    # already; issue CYCLE_GOAL explicitly on odd reach episodes to smoke-test
    # the goal-switch hook, then hand back to caller for pipeline.reset().
    cycle_goal_fired = False
    if is_reach and (episode_idx % 2 == 1):
        commands = ["[CYCLE_GOAL]"]
        pipeline.exec.handle_commands(commands, current_dof=pipeline.env.dof_pos, ready_pose=pipeline._ready_pose)
        inner_policy.post_step_callback(commands)
        cycle_goal_fired = True

    pipeline.policy.get_observation = orig_get_observation  # restore

    recorder_stats = pipeline.recorder.stats()

    metrics = {
        "episode": episode_idx,
        "episode_steps": episode_steps,
        "wall_clock_s": wall_elapsed,
        "wall_clock_hz": episode_steps / wall_elapsed if wall_elapsed > 0 else None,
        "recorder_stats": recorder_stats,
        "raw_action_norm": _pstats(raw_action_norms),
        "dof_excursion_rad": _pstats(dof_excursion),
        "wrist_excursion_m": _pstats(wrist_excursion),
        "cycle_goal_fired": cycle_goal_fired,
        "object_pos_available": object_pos_available,
    }
    if is_reach:
        if goal_wrist_error_abs:
            arr = np.asarray(goal_wrist_error_abs)  # (T, 6)
            metrics["goal_wrist_error_abs_m"] = {
                "mean_per_axis": arr.mean(axis=0).tolist(),
                "max_per_axis": arr.max(axis=0).tolist(),
                "mean_norm": float(np.linalg.norm(arr, axis=1).mean()),
                "max_norm": float(np.linalg.norm(arr, axis=1).max()),
            }
        else:
            metrics["goal_wrist_error_abs_m"] = None
    if object_pos_samples:
        arr = np.asarray(object_pos_samples)
        metrics["object_pos_range_m"] = (arr.max(axis=0) - arr.min(axis=0)).tolist()
    else:
        metrics["object_pos_note"] = "env.object_pos unavailable (no object body in this MuJoCo scene)"

    return metrics


def assemble_video(video_writer, out_path: Path, fps: int):
    import imageio.v2 as imageio

    frames = video_writer["frames"]
    if not frames:
        return False
    try:
        writer = imageio.get_writer(str(out_path), fps=fps, macro_block_size=None)
        for fp in frames:
            writer.append_data(imageio.imread(fp))
        writer.close()
        return True
    except Exception as e:  # video is evidence, never load-bearing
        logger.warning(f"[sim2sim_eval] video assembly failed ({e}); metrics unaffected")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Registered RlPipelineCfg name")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--video", action="store_true", help="Attempt offscreen MuJoCo video capture")
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--video-every-n", type=int, default=2, help="Capture 1 frame every N sim steps")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_reach = "reach" in args.config
    freq = None  # resolved after cfg parse

    # Build once; episode_steps needs freq, so parse a throwaway cfg first.
    probe_cfg = ConfigManager(config_name=args.config).get_cfg()
    freq = probe_cfg.policy.freq
    episode_steps = int(round(args.episode_seconds * freq))

    pipeline, cfg = build_pipeline(args.config, episode_steps=episode_steps)

    # Deploy-default startup: ramp to READY pose, hold FROZEN (mirrors
    # scripts/run_pipeline.py's main()).
    wbc_cfg = getattr(cfg, "wbc", None)
    if getattr(wbc_cfg, "startup_ready_pose", False):
        pipeline.startup()

    all_metrics = []
    tmp_frame_dir = out_dir / "_frames_tmp"
    video_paths = []
    for ep in range(args.episodes):
        video_writer = None
        if args.video:
            ep_tmp = tmp_frame_dir / f"ep{ep}"
            ep_tmp.mkdir(parents=True, exist_ok=True)
            video_writer = {
                "tmp_dir": ep_tmp,
                "frames": [],
                "width": args.video_width,
                "height": args.video_height,
                "render_failed": False,
            }

        logger.info(f"=== episode {ep}/{args.episodes} ({episode_steps} steps @ {freq}Hz) ===")
        metrics = run_episode(
            pipeline, cfg, ep, episode_steps, is_reach,
            video_writer=video_writer, video_every_n=args.video_every_n,
        )

        if video_writer is not None:
            if video_writer["frames"]:
                video_path = out_dir / f"{args.config}_ep{ep}.mp4"
                ok = assemble_video(video_writer, video_path, fps=args.video_fps)
                if ok:
                    video_paths.append(str(video_path))
                    metrics["video_path"] = str(video_path)
                else:
                    metrics["video_path"] = None
            else:
                metrics["video_path"] = None
                metrics["video_note"] = "no frames captured (GL context / MuJoCo Renderer failed)"

        all_metrics.append(metrics)
        pipeline.reset()

    pipeline._teardown_deploy_resources()

    # Cleanup temp frame PNGs (video is already assembled).
    if tmp_frame_dir.exists():
        import shutil

        shutil.rmtree(tmp_frame_dir, ignore_errors=True)

    run_summary = {
        "config": args.config,
        "episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
        "episode_steps": episode_steps,
        "freq_hz": freq,
        "is_reach": is_reach,
        "video_paths": video_paths,
        "per_episode": all_metrics,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(run_summary, f, indent=2, default=str)
    logger.info(f"Wrote {metrics_path}")

    write_markdown_summary(run_summary, out_dir / "summary.md")
    logger.info(f"Wrote {out_dir / 'summary.md'}")


def write_markdown_summary(run_summary: dict, out_path: Path):
    config = run_summary["config"]
    is_reach = run_summary["is_reach"]
    lines = [
        f"# sim2sim eval — `{config}`",
        "",
        f"episodes={run_summary['episodes']}, episode_seconds={run_summary['episode_seconds']}, "
        f"episode_steps={run_summary['episode_steps']}, freq={run_summary['freq_hz']}Hz",
        "",
    ]
    if is_reach:
        lines += [
            "Action side runs through the frozen masked-mimic WBC ONNX (integral wrist-command "
            "mode; faithful two-stage control law for arms/wrists, legs unmasked — see "
            "H1_2ReachTeacherOnnxPolicy module docstring). Finger actuators absent on this MuJoCo asset.",
            "",
        ]
    else:
        lines += [
            "Action side runs through the frozen masked-mimic WBC ONNX (faithful two-stage control "
            "law for arms/wrists; legs unmasked). Finger actuators absent on this MuJoCo asset.",
            "",
        ]

    header = [
        "episode", "wall_hz", "raw_action_norm(mean/p95/max)",
        "dof_excursion_rad(mean/p95/max)", "wrist_excursion_m(mean/p95/max)",
    ]
    if is_reach:
        header.append("goal_wrist_error_norm(mean/max)")
    header.append("video")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))

    for m in run_summary["per_episode"]:
        def fmt(d):
            if d is None:
                return "n/a"
            return f"{d['mean']:.4f}/{d['p95']:.4f}/{d['max']:.4f}" if d["mean"] is not None else "n/a"

        row = [
            str(m["episode"]),
            f"{m['wall_clock_hz']:.1f}" if m["wall_clock_hz"] else "n/a",
            fmt(m["raw_action_norm"]),
            fmt(m["dof_excursion_rad"]),
            fmt(m["wrist_excursion_m"]),
        ]
        if is_reach:
            gwe = m.get("goal_wrist_error_abs_m")
            row.append(f"{gwe['mean_norm']:.4f}/{gwe['max_norm']:.4f}" if gwe else "n/a")
        row.append(Path(m["video_path"]).name if m.get("video_path") else "n/a")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Recorder Hz/latency (cumulative snapshot, last episode)")
    if run_summary["per_episode"]:
        rec = run_summary["per_episode"][-1]["recorder_stats"]
        lines.append("| component | count | mean_hz | p95_latency_ms |")
        lines.append("|---|---|---|---|")
        for comp, st in rec.items():
            mean_hz = f"{st['mean_hz']:.1f}" if st.get("mean_hz") else "n/a"
            p95 = f"{st['p95_latency_ms']:.3f}" if st.get("p95_latency_ms") else "n/a"
            lines.append(f"| {comp} | {st['count']} | {mean_hz} | {p95} |")

    if not is_reach:
        lines.append("")
        lines.append(
            "## lift box proxy: `env.object_pos` unavailable on this MuJoCo scene "
            "(h1_2_box_feet.xml has no cube/box body) — not faked, see per-episode "
            "`object_pos_note` in metrics.json."
        )

    out_path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
