"""Native RoboJuDo pipeline for finite Imprint deployment sessions."""

from typing import Any

import robojudo.deployment
from robojudo.pipeline import Pipeline, pipeline_registry
from robojudo.pipeline.pipeline_cfgs import DeploymentPipelineCfg


@pipeline_registry.register
class DeploymentPipeline(Pipeline):
    cfg: DeploymentPipelineCfg

    def __init__(self, cfg: DeploymentPipelineCfg):
        super().__init__(cfg)
        if cfg.steps <= 0:
            raise ValueError("deployment steps must be positive")
        adapter_class: type = getattr(robojudo.deployment, cfg.adapter_type)
        self.adapter = adapter_class()
        self.session = self.adapter.build(cfg)
        self.result: Any = None

    def prepare(self):
        """The bootstrap owns any simulator- or hardware-specific preparation."""

    def step(self):
        raise RuntimeError("DeploymentPipeline is finite; call run_to_completion()")

    def run_to_completion(self):
        self.result = self.session.run(self.cfg.steps)
        return self.result
