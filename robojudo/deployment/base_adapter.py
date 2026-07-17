"""Deployment adapter interface owned by RoboJuDo."""

from abc import ABC, abstractmethod
from typing import Any


class DeploymentAdapter(ABC):
    """Build a finite rollout session without importing a simulator eagerly."""

    @abstractmethod
    def build(self, cfg: Any) -> Any:
        """Return an object exposing ``run(steps)``."""
        raise NotImplementedError
