"""Registered deployment adapters for finite external-policy rollouts."""

from robojudo.utils.module_registry import Registry

from .base_adapter import DeploymentAdapter

deployment_adapter_registry = Registry(
    package="robojudo.deployment", base_class=DeploymentAdapter
)

__all__ = ["DeploymentAdapter", "deployment_adapter_registry"]


def __getattr__(name: str) -> type[DeploymentAdapter]:
    try:
        adapter_class = deployment_adapter_registry.get(name)
    except Exception as error:
        raise AttributeError(f"module {__name__} has no attribute {name}") from error
    globals()[name] = adapter_class
    return adapter_class


deployment_adapter_registry.add("ImprintNewtonAdapter", ".imprint_adapter")
deployment_adapter_registry.add("ImprintMujocoAdapter", ".imprint_adapter")
