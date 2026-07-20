"""Hardware-free mock policy for headless deploy-harness testing.

Produces a deterministic, frame-dependent action WITHOUT loading any ONNX/JIT
model, so the full pipeline (env -> ctrl -> policy -> WBC state machine -> env)
can be exercised on the DummyEnv with no GPU and no checkpoint.

The action is a small sinusoid keyed on the internal ``_frame`` counter, so the
pd_target *changes* whenever the policy frame advances.  This lets tests prove
that FREEZE holds the frame (constant target) while single-step / burst advance
it exactly the expected number of times.
"""

import logging

import numpy as np

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import PolicyCfg

logger = logging.getLogger(__name__)


@policy_registry.register
class MockPolicy(Policy):
    cfg_policy: PolicyCfg

    def __init__(self, cfg_policy: PolicyCfg, device: str = "cpu"):
        super().__init__(cfg_policy=cfg_policy, device=device)
        self._frame = 0
        self._paused = False
        logger.info(f"[MockPolicy] initialized (num_actions={self.num_actions})")

    def reset(self):
        self._frame = 0
        self._paused = False
        self.last_action = np.zeros(self.num_actions)

    def reset_alignment(self):
        logger.info("[MockPolicy] reset_alignment (obs re-sync)")

    def post_step_callback(self, commands=None):
        if not self._paused:
            self._frame += 1

    def get_observation(self, env_data, ctrl_data):
        obs = np.asarray(env_data.dof_pos, dtype=np.float32)
        extras = {"CALLBACK": [], "hand_pose": None}
        return obs, extras

    def get_action(self, obs):
        # Deterministic, frame-dependent action (small, safe amplitude).
        amp = 0.05
        phase = self._frame * 0.1
        action = amp * np.sin(phase + np.arange(self.num_actions) * 0.05)
        self.last_action = action.astype(np.float32)
        return self.last_action

    def get_init_dof_pos(self):
        return self.default_pos.copy()

    @property
    def current_frame(self) -> int:
        return self._frame
