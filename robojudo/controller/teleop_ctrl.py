"""Whole-body teleop controller for RoboJuDo.

imprint integration (additive) -- this module is NOT part of native RoboJuDo.
It bridges an ``imprint.robojudo.teleop`` provider (a teleop source + live
mink retargeter) into the RoboJuDo controller stack.  Each tick it emits a
ctrl_data dict of retargeted references that the ProtoMotions tracker policy
reads through its LiveRefSource seam.

The imprint provider is imported LAZILY inside ``__init__`` so that merely
importing this module (which the controller registry may do to introspect the
class) does not require the imprint package to be installed.
"""

import logging

from robojudo.controller import Controller, ctrl_registry
from robojudo.controller.ctrl_cfgs import TeleopCtrlCfg

logger = logging.getLogger(__name__)


@ctrl_registry.register
class TeleopCtrl(Controller):
    cfg_ctrl: TeleopCtrlCfg

    def __init__(self, cfg_ctrl: TeleopCtrlCfg, env=None, device="cpu"):
        super().__init__(cfg_ctrl=cfg_ctrl, env=env, device=device)

        # Lazy import: only pull in the imprint teleop stack when a TeleopCtrl
        # is actually instantiated (keeps native RoboJuDo import-clean).
        from imprint.robojudo.teleop.provider import build_teleop_provider

        self._provider = build_teleop_provider(
            source_kind=self.cfg_ctrl.source,
            motion_path=self.cfg_ctrl.motion_path,
            motion_index=self.cfg_ctrl.motion_index,
            pool=self.cfg_ctrl.pool,
            index=self.cfg_ctrl.index,
            loop=self.cfg_ctrl.loop,
            speed=self.cfg_ctrl.speed,
            zmq_host=self.cfg_ctrl.zmq_host,
            zmq_port=self.cfg_ctrl.zmq_port,
            legs=self.cfg_ctrl.legs,
            iters=self.cfg_ctrl.iters,
            retarget=self.cfg_ctrl.retarget,
        )
        logger.info(
            f"[TeleopCtrl] Initialized (source={self.cfg_ctrl.source}, "
            f"legs={self.cfg_ctrl.legs})."
        )

    def reset(self):
        self._provider.source.reset()

    def get_data(self):
        return self._provider.get_ctrl_data()

    def process_triggers(self, ctrl_data):
        """Map the provider's ``buttons`` dict onto RoboJuDo COMMANDS."""
        commands = []
        buttons = {}
        if ctrl_data is not None:
            buttons = ctrl_data.get("buttons", {}) or {}
        if buttons.get("reset"):
            commands.append("[MOTION_RESET]")
        if buttons.get("fade_out"):
            commands.append("[MOTION_FADE_OUT]")
        if buttons.get("shutdown"):
            commands.append("[SHUTDOWN]")
        return ctrl_data, commands
