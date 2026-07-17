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

        # NOTE: relative/centered teleop is NOT a TeleopCtrlCfg field (that
        # dataclass is upstream RoboJuDo, kept pristine). It defaults to
        # disabled here and is toggled after construction via
        # ``self._provider.set_relative_enabled(...)`` -- see
        # ``imprint.robojudo.gated_inference`` (--teleop-relative wiring), which
        # calls it once the pipeline/provider has been built. Plain instance
        # attribute (not a cfg field) so callers can also flip it directly.
        self.relative_enabled: bool = False

        # NOTE: the live-source staleness guard's timeout is NOT a
        # TeleopCtrlCfg field either (same reasoning as relative_enabled
        # above -- that dataclass is upstream RoboJuDo, kept pristine). It
        # defaults to today's 0.4s here and is overridden after construction
        # via ``self._provider.set_stale_timeout_s(...)`` -- see
        # ``imprint.robojudo.gated_inference`` (--teleop-stale-timeout
        # wiring), which calls it once the pipeline/provider has been built.
        # Plain instance attribute (not a cfg field) so callers can also
        # flip it directly.
        self.stale_timeout_s: float = 0.4

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
            stale_timeout_s=self.stale_timeout_s,
            legs=self.cfg_ctrl.legs,
            iters=self.cfg_ctrl.iters,
            retarget=self.cfg_ctrl.retarget,
            relative_enabled=self.relative_enabled,
        )
        logger.info(
            f"[TeleopCtrl] Initialized (source={self.cfg_ctrl.source}, "
            f"legs={self.cfg_ctrl.legs}, relative_enabled={self.relative_enabled})."
        )

    def reset(self):
        self._provider.source.reset()

    def get_data(self):
        # NOTE: TeleopCtrl is a plain Controller (not ControllerHook), so
        # CtrlManager.get_ctrl_data(env_data)'s env_data does NOT reach here --
        # only ControllerHook.get_data_with_hook receives it. ``self.env`` is
        # however stored at construction time (Controller.__init__), so the
        # measured dof_pos (the truest "robot's current pose" anchor for
        # relative-teleop align/clutch) is still available via that seam.
        robot_dof_pos = getattr(self.env, "dof_pos", None) if self.env is not None else None
        return self._provider.get_ctrl_data(robot_dof_pos=robot_dof_pos)

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
