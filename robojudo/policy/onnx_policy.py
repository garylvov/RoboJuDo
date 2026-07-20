"""Generic ONNX policy runner for RoboJuDo.

Loads an arbitrary ONNX-exported checkpoint via ``onnxruntime`` and threads an
optional recurrent (LSTM) hidden/cell state across steps, exactly like the
jit-based :class:`Policy` base class does for the actor MLP -- but for ONNX
graphs that were exported with an explicit ``h_in``/``c_in`` -> ``h_out``/
``c_out`` recurrent contract (as ``legacy_actor_critic_graph`` exports do; see
``meta.json`` sidecars produced by the imprint ONNX exporter).

Mirrors :class:`~robojudo.policy.mock_policy.MockPolicy`'s interface (same
``reset`` / ``post_step_callback`` / ``get_observation`` / ``get_action``
contract) so it drops into the same deploy pipeline / WBC state machine
without any pipeline-side changes.

This base class intentionally does NOT know how to build a task-specific
observation vector -- ``get_observation`` here is a generic fallback (raw
``dof_pos``, zero-padded to ``obs_dim``) suitable only for smoke-testing the
ONNX plumbing itself. Concrete deployments (e.g. the H1_2 lift-teacher) should
subclass and override ``get_observation`` with the real per-task obs mapping.
"""

import logging

import numpy as np
import onnxruntime as ort

from robojudo.policy import Policy, policy_registry
from robojudo.policy.policy_cfgs import OnnxPolicyCfg

logger = logging.getLogger(__name__)


@policy_registry.register
class OnnxPolicy(Policy):
    """ONNX inference with explicit LSTM h/c state threading + EP fallback."""

    cfg_policy: OnnxPolicyCfg

    def __init__(self, cfg_policy: OnnxPolicyCfg, device: str = "cpu"):
        # Skip the jit-load path in Policy.__init__ -- we manage our own
        # onnxruntime session below.
        cfg_policy_new = cfg_policy.model_copy()
        cfg_policy_new.disable_autoload = True
        super().__init__(cfg_policy=cfg_policy_new, device=device)

        onnx_path = cfg_policy.policy_file
        providers = self._resolve_providers(cfg_policy.providers)
        logger.info(f"[OnnxPolicy] loading {onnx_path} with providers={providers}")
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:  # pragma: no cover - EP-specific failures vary by build
            if providers != ["CPUExecutionProvider"]:
                logger.warning(f"[OnnxPolicy] providers={providers} failed ({e}); falling back to CPU")
                providers = ["CPUExecutionProvider"]
                self.session = ort.InferenceSession(onnx_path, providers=providers)
            else:
                raise
        self.active_providers = self.session.get_providers()
        logger.info(f"[OnnxPolicy] active providers: {self.active_providers}")

        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.is_recurrent = cfg_policy.is_recurrent
        self.rnn_hidden_dim = cfg_policy.rnn_hidden_dim
        self.rnn_num_layers = cfg_policy.rnn_num_layers

        self.obs_input_name = cfg_policy.obs_input_name if cfg_policy.obs_input_name in self.input_names else self.input_names[0]
        self.action_output_name = (
            cfg_policy.action_output_name if cfg_policy.action_output_name in self.output_names else self.output_names[0]
        )
        self.h_in_name = cfg_policy.h_in_name
        self.c_in_name = cfg_policy.c_in_name
        self.h_out_name = cfg_policy.h_out_name
        self.c_out_name = cfg_policy.c_out_name

        self._h = None
        self._c = None
        self.reset()

    @staticmethod
    def _resolve_providers(requested: list[str]) -> list[str]:
        available = ort.get_available_providers()
        providers = [p for p in requested if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        if "CPUExecutionProvider" not in providers:
            providers.append("CPUExecutionProvider")
        return providers

    def reset(self):
        """Zero-on-reset: LSTM state and last_action both zeroed."""
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        if self.is_recurrent:
            self._h = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_dim), dtype=np.float32)
            self._c = np.zeros((self.rnn_num_layers, 1, self.rnn_hidden_dim), dtype=np.float32)

    def reset_alignment(self):
        return

    def post_step_callback(self, commands: list[str] | None = None):
        for cmd in commands or []:
            if cmd == "[MOTION_RESET]":
                self.reset()

    def get_observation(self, env_data, ctrl_data) -> tuple[np.ndarray, dict]:
        """Generic fallback obs (dof_pos, zero-padded). Override for real deployments."""
        dof_pos = np.asarray(env_data.dof_pos, dtype=np.float32)
        obs = np.zeros(self.cfg_policy.obs_dim, dtype=np.float32)
        n = min(len(dof_pos), len(obs))
        obs[:n] = dof_pos[:n]
        extras = {"CALLBACK": [], "hand_pose": None}
        return obs, extras

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ort_inputs = {self.obs_input_name: np.expand_dims(obs, axis=0).astype(np.float32)}
        if self.is_recurrent:
            ort_inputs[self.h_in_name] = self._h
            ort_inputs[self.c_in_name] = self._c

        ort_outputs = self.session.run(self.output_names, ort_inputs)
        out_by_name = dict(zip(self.output_names, ort_outputs))

        actions = np.asarray(out_by_name[self.action_output_name]).squeeze().astype(np.float32)
        if self.is_recurrent:
            self._h = np.asarray(out_by_name[self.h_out_name], dtype=np.float32)
            self._c = np.asarray(out_by_name[self.c_out_name], dtype=np.float32)

        actions = (1 - self.action_beta) * self.last_action + self.action_beta * actions
        self.last_action = actions.copy()

        processed = actions
        if self.action_clip is not None:
            processed = np.clip(processed, -self.action_clip, self.action_clip)
        processed = processed * self.action_scale
        return processed

    def get_init_dof_pos(self) -> np.ndarray:
        return self.default_pos.copy()
