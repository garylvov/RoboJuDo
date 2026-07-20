"""Smoke test: H1_2 reach-teacher ONNX + frozen masked-mimic WBC chain, on the
NEWTON backend (``h1_2_newton_reach_deploy``), 2 episodes, standing check +
Ability-hand finger-mapping demo + a saved camera frame.

Why a standalone script instead of ``scripts/run_pipeline.py``: the reach-
teacher policy's own action space (12-dim, wbc_reach conditioning only) has NO
finger head (unlike the lift-teacher's 24-dim action -- see
``h1_2_newton_lift_deploy``'s docstring), so there is nothing in the ONNX
policy path to exercise ``NewtonEnv``'s new Ability-hand finger mapping with.
This script drives the SAME pipeline (``RlPipeline`` + ``h1_2_newton_reach_deploy``,
identical WBC-chain / ONNX / scripted-schedule wiring ``run_pipeline.py`` would
use) but additionally wraps ``env.step`` with a scripted sinusoidal
``hand_pose`` command each control step -- exercising the EXACT same
``NewtonEnv._apply_hand_pose`` / 4-bar-coupling code path the lift-teacher's
``ability_fingers`` action head drives, independent of what the reach ONNX
itself outputs (which never touches ``hand_pose``).

Usage::

    CUDA_VISIBLE_DEVICES=<gpu> python scripts/newton_reach_smoke_test.py \\
        --steps-per-episode 150 --out-dir /tmp/newton_reach_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import robojudo.pipeline  # noqa: F401 - registers pipeline classes
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo.newton_reach_smoke_test")


def _scripted_hand_pose(t: float) -> np.ndarray:
    """12-dim [left(6), right(6)] scripted open/close command, per-hand order
    [index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2] (see
    NewtonEnv module header). A slow sinusoid sweeps the 4 fingers + thumb
    flexor through their full [0, 1.74] range (thumb rotator stays near 0 --
    its range is negative and less relevant to a visible "grasp" demo)."""
    phase = 0.5 * (1.0 - np.cos(2.0 * np.pi * 0.15 * t))  # 0->1->0, ~6.7s period
    finger = phase * 1.6  # index/middle/ring/pinky q1
    thumb_flex = phase * 1.6  # thumb_q2
    thumb_rot = -0.2 * phase  # thumb_q1 (small abduction sweep, stays in [-1.74, 0])
    one_hand = np.array([finger, finger, finger, finger, thumb_rot, thumb_flex], dtype=np.float32)
    return np.concatenate([one_hand, one_hand])


def _install_scripted_hand_pose(env):
    """Wrap ``env.step`` ONCE so every call injects the scripted sinusoidal
    hand_pose (see ``_scripted_hand_pose``) regardless of what the caller
    (the RlPipeline, via the reach-teacher policy's ``extras.get("hand_pose")``,
    always None for this policy -- see module docstring) passes. Exercises the
    SAME ``NewtonEnv.step()``/``_apply_hand_pose`` code path the lift-teacher's
    ``ability_fingers`` action head drives. Reads a mutable ``env._smoke_t``
    counter the caller advances each control step."""
    if hasattr(env, "_orig_step"):
        return  # already installed
    env._orig_step = env.step
    env._smoke_t = 0

    def _stepped(pd_target, hand_pose=None):
        return env._orig_step(pd_target, hand_pose=_scripted_hand_pose(float(env._smoke_t)))

    env.step = _stepped


def run_episode(pipeline: RlPipeline, episode_idx: int, steps: int, out_dir: Path) -> dict:
    env = pipeline.env
    base_z = []
    finger_log = []

    for t in range(steps):
        env._smoke_t = t
        pipeline.step()

        base_z.append(float(env.base_pos[2]) if env.base_pos is not None else float("nan"))
        if env.has_fingers and (t % max(steps // 6, 1) == 0):
            fjp = env.finger_joint_pos
            finger_log.append({"t": t, "finger_joint_pos": [round(float(x), 4) for x in fjp]})
            logger.info(f"[episode {episode_idx}] t={t} finger_joint_pos={fjp}")

    base_z = np.asarray(base_z)
    standing = bool(np.all(base_z > 0.5))  # H1_2 standing pelvis ~1.0m; fallen collapses well below 0.5m
    summary = {
        "episode": episode_idx,
        "steps": steps,
        "base_z_min": float(base_z.min()),
        "base_z_max": float(base_z.max()),
        "base_z_final": float(base_z[-1]),
        "standing_verdict": "STANDING" if standing else "FELL",
        "has_fingers": bool(env.has_fingers),
        "finger_log": finger_log,
    }

    if env.has_fingers and len(finger_log) >= 2:
        delta = np.abs(np.asarray(finger_log[-1]["finger_joint_pos"]) - np.asarray(finger_log[0]["finger_joint_pos"]))
        summary["finger_actuated"] = bool(np.any(delta > 0.05))
        summary["finger_max_delta"] = float(delta.max())

    frame_path = out_dir / f"episode{episode_idx}_camera_native_424x240.png"
    frame_ds_path = out_dir / f"episode{episode_idx}_camera_sim_212x120.png"
    try:
        env.save_camera_frame(str(frame_path))
        env.save_camera_frame(str(frame_ds_path), downsample=True)
        summary["camera_frame"] = str(frame_path)
        summary["camera_frame_sim"] = str(frame_ds_path)
    except Exception as e:  # camera is evidence, never load-bearing
        logger.warning(f"[episode {episode_idx}] camera frame failed: {e}")
        summary["camera_frame"] = None

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="h1_2_newton_reach_deploy")
    parser.add_argument("--steps-per-episode", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--out-dir", default="/tmp/newton_reach_smoke")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_manager = ConfigManager(config_name=args.config)
    cfg = config_manager.get_cfg()

    t_build0 = time.time()
    pipeline = RlPipeline(cfg=cfg)
    logger.info(f"Pipeline built in {time.time() - t_build0:.1f}s (env={type(pipeline.env).__name__})")
    _install_scripted_hand_pose(pipeline.env)

    wbc_cfg = getattr(cfg, "wbc", None)
    if wbc_cfg is not None and getattr(wbc_cfg, "startup_ready_pose", False):
        pipeline.startup()

    episode_summaries = []
    for ep in range(args.episodes):
        logger.info(f"===== episode {ep} =====")
        summary = run_episode(pipeline, ep, args.steps_per_episode, out_dir)
        episode_summaries.append(summary)
        logger.info(f"episode {ep} summary: {json.dumps(summary, indent=2, default=str)[:2000]}")
        if ep < args.episodes - 1:
            pipeline.reset()
            if wbc_cfg is not None and getattr(wbc_cfg, "startup_ready_pose", False):
                pipeline.startup()

    pipeline.env.shutdown()

    report = {"config": args.config, "episodes": episode_summaries}
    report_path = out_dir / "smoke_test_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Wrote report to {report_path}")

    for s in episode_summaries:
        print(
            f"episode {s['episode']}: {s['standing_verdict']} "
            f"(base_z min={s['base_z_min']:.3f} max={s['base_z_max']:.3f} final={s['base_z_final']:.3f}) "
            f"finger_actuated={s.get('finger_actuated')} finger_max_delta={s.get('finger_max_delta')}"
        )


if __name__ == "__main__":
    main()
