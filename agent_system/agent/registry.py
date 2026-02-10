# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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