"""Lazy adapters for Imprint's Newton and MuJoCo deployment backends."""

import importlib
from typing import Any

from robojudo.deployment import DeploymentAdapter, deployment_adapter_registry


def _resolve(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("bootstrap must use the 'module:callable' form")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"deployment bootstrap {spec!r} is not callable")
    return factory


class _ImprintAdapter(DeploymentAdapter):
    backend_kind: str

    def build(self, cfg: Any) -> Any:
        session = _resolve(cfg.bootstrap)(**dict(cfg.bootstrap_kwargs))
        run = getattr(session, "run", None)
        if not callable(run):
            raise TypeError("deployment bootstrap must return an object exposing run(steps)")
        backend = getattr(session, "backend", None)
        capabilities = getattr(backend, "capabilities", None)
        actual_kind = getattr(capabilities, "kind", None)
        if actual_kind != self.backend_kind:
            raise TypeError(
                f"{type(self).__name__} requires backend kind {self.backend_kind!r}, "
                f"got {actual_kind!r}"
            )
        return session


@deployment_adapter_registry.register
class ImprintNewtonAdapter(_ImprintAdapter):
    """Registered RoboJuDo boundary for the owned IsaacLab/Newton shim."""

    backend_kind = "newton"


@deployment_adapter_registry.register
class ImprintMujocoAdapter(_ImprintAdapter):
    """Registered RoboJuDo boundary for an owned wrapper around MujocoEnv."""

    backend_kind = "mujoco"
