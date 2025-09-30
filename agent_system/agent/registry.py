from __future__ import annotations
"""
Agent registry.
"""
from typing import Dict, Callable, List


class AgentRegistry:
    _REGISTRY: Dict[str, Callable[..., "BaseAgent"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(agent_cls: Callable[..., "BaseAgent"]):
            if name in cls._REGISTRY:
                raise ValueError(f"Agent '{name}' already registered.")
            cls._REGISTRY[name] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._REGISTRY:
            raise KeyError(f"Unknown agent '{name}'. Registered: {list(cls._REGISTRY)}")
        return cls._REGISTRY[name](**kwargs)

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._REGISTRY)