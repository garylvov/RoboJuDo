"""Scripted controller — emits a preprogrammed command schedule by step.

Headless-friendly stand-in for keyboard/joystick controllers (which need an X
display / a physical gamepad).  Lets ``scripts/run_pipeline.py`` drive the WBC
execution state machine deterministically in CI/harness runs without hardware.
"""

import logging

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import ScriptedCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class ScriptedCtrl(Controller):
    cfg_ctrl: ScriptedCtrlCfg

    def __init__(self, cfg_ctrl: ScriptedCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)
        # schedule keys may be ints or str (from JSON/config) -> normalize to int
        self.schedule = {int(k): list(v) for k, v in cfg_ctrl.schedule.items()}
        self.reset()

    def reset(self):
        self._t = 0

    def get_data(self):
        return {"scripted_step": self._t}

    def process_triggers(self, ctrl_data):
        commands = list(self.schedule.get(self._t, []))
        if commands:
            logger.info(f"[ScriptedCtrl] step {self._t} -> {commands}")
        self._t += 1
        return ctrl_data, commands
