# Fix OMP perfmance issue on ARM platform (Jetson)
import os
import platform

if platform.machine().startswith("aarch64"):
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import logging
import time

import robojudo.pipeline
from robojudo.config.config_manager import ConfigManager
from robojudo.pipeline.pipeline_cfgs import RlPipelineCfg
from robojudo.pipeline.rl_pipeline import RlPipeline

logger = logging.getLogger("robojudo")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="g1",
        help="Name of the config class to use",
    )
    parser.add_argument(
        "--config-module",
        action="append",
        default=[],
        help="Import a module that registers external RoboJuDo configs",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override the finite DeploymentPipeline step count",
    )
    parser.add_argument(
        "--bootstrap",
        default=None,
        help="Override a DeploymentPipeline bootstrap (module:callable)",
    )
    parser.add_argument(
        "--bootstrap-kwargs-json",
        default=None,
        help="JSON object passed to the deployment bootstrap",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Stop the RlPipeline loop after this many steps (test/harness use)",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    for module_name in args.config_module:
        importlib.import_module(module_name)
    logger.info(f"Using config: {args.config}")
    config_manager = ConfigManager(config_name=args.config)

    cfg: RlPipelineCfg = config_manager.get_cfg()
    if args.bootstrap is not None or args.bootstrap_kwargs_json is not None:
        if getattr(cfg, "pipeline_type", None) != "DeploymentPipeline":
            raise ValueError("deployment bootstrap overrides require DeploymentPipeline")
        if args.bootstrap is not None:
            cfg.bootstrap = args.bootstrap
        if args.bootstrap_kwargs_json is not None:
            kwargs = json.loads(args.bootstrap_kwargs_json)
            if not isinstance(kwargs, dict):
                raise ValueError("--bootstrap-kwargs-json must decode to an object")
            cfg.bootstrap_kwargs = kwargs
    if args.steps is not None:
        if getattr(cfg, "pipeline_type", None) != "DeploymentPipeline":
            raise ValueError("--steps is only valid for DeploymentPipeline configs")
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        cfg.steps = args.steps

    pipeline_type = cfg.pipeline_type

    pipeline_class: type[RlPipeline] = getattr(robojudo.pipeline, pipeline_type)
    logger.info(f"Using pipeline: {pipeline_type} -> {pipeline_class}")

    pipeline = pipeline_class(cfg=cfg)

    run_to_completion = getattr(pipeline, "run_to_completion", None)
    if run_to_completion is not None:
        run_to_completion()
        return

    # Deploy-default startup flow: move to READY pose with NO policy running,
    # then hold frozen.  Policy engagement is a deliberate, user-triggered step
    # ([RESUME_POLICY] / stepped commands).  Opt-in via cfg.wbc.startup_ready_pose.
    wbc_cfg = getattr(cfg, "wbc", None)
    startup = getattr(pipeline, "startup", None)
    if startup is not None and wbc_cfg is not None and getattr(wbc_cfg, "startup_ready_pose", False):
        logger.warning("Deploy startup: ready-pose-first, policy engaged on command")
        startup()
    elif not cfg.env.is_sim:
        pipeline.prepare()
    elif getattr(pipeline, "_has_default_pose_mode", False):
        pipeline._set_default_pose_mode(True)
        logger.warning("Sim mode — holding default pose, press R to start motion")

    step_count = 0
    while True:
        time_start = time.time()
        pipeline.step()
        step_count += 1
        if args.max_steps is not None and step_count >= args.max_steps:
            logger.warning(f"Reached --max-steps={args.max_steps}, stopping")
            teardown = getattr(pipeline, "_teardown_deploy_resources", None)
            if teardown is not None:
                teardown()
            break
        time_end = time.time()
        time_diff = time_end - time_start

        # keep the pipeline running at the desired frequency
        if not cfg.run_fullspeed:
            time_diff = pipeline.dt - time_diff
            if time_diff > 0:
                time.sleep(time_diff)
            else:
                if not cfg.env.is_sim:
                    logger.error(f"Warning: frame drop -> {time_diff}")
                    if time_diff < -0.2:
                        logger.critical("Exiting due to excessive frame drop")
                        pipeline.env.shutdown()
                        time.sleep(10)
                        break


if __name__ == "__main__":
    main()
