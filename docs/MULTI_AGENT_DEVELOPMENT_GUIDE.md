# Multi-Agent LLM Systems Development Guide

This guide provides comprehensive instructions for developing custom multi-agent LLM systems using the Dr.MAS framework. It covers agent registration, orchestra orchestration, configuration, and integration with the training pipeline.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Core Components](#core-components)
- [Developing Custom Agents](#developing-custom-agents)
  - [Step 1: Understanding the Base Agent](#step-1-understanding-the-base-agent)
  - [Step 2: Creating a New Agent](#step-2-creating-a-new-agent)
  - [Step 3: Registering the Agent](#step-3-registering-the-agent)
  - [Step 4: Agent Best Practices](#step-4-agent-best-practices)
- [Developing Custom Orchestras](#developing-custom-orchestras)
  - [Step 1: Understanding the Base Orchestra](#step-1-understanding-the-base-orchestra)
  - [Step 2: Creating a New Orchestra](#step-2-creating-a-new-orchestra)
  - [Step 3: Registering the Orchestra](#step-3-registering-the-orchestra)
  - [Step 4: Orchestra Patterns](#step-4-orchestra-patterns)
- [Configuration Guide](#configuration-guide)
  - [Agent Configuration](#agent-configuration)
  - [Per-Agent Parameter Overrides](#per-agent-parameter-overrides)
  - [Orchestra Configuration](#orchestra-configuration)
- [Running Your Multi-Agent System](#running-your-multi-agent-system)
- [Summary](#summary)

---

## Architecture Overview

Dr.MAS follows a modular architecture for multi-agent LLM systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MultiAgentTrajectoryCollector                │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │                    Orchestra                        │  │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  │   │
│  │  │  │ Agent 1  │  │ Agent 2  │  │ Agent N  │  ...    │  │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘         │  │   │
│  │  │        │              │              │             │  │   │
│  │  │        └──────────────┼──────────────┘             │  │   │
│  │  │                       ▼                            │  │   │
│  │  │              LLM Worker Groups                     │  │   │
│  │  │       (Shared or Dedicated per Agent)              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│                        Environment                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Concepts

1. **Agent**: A specialized LLM-based component with a specific role (e.g., Solver, Verifier, Search)
2. **Orchestra**: Coordinates the execution flow of multiple agents
3. **Worker Group (WG)**: An LLM inference engine that can be shared or dedicated to agents
4. **Agent Registry**: A centralized registry for agent classes using decorator-based registration
5. **Team Context**: Shared context that accumulates agent outputs for inter-agent communication

---

## Core Components

### Directory Structure

```
agent_system/
├── agent/
│   ├── registry.py          # Agent registry implementation
│   ├── utils.py              # Utility functions (projections, config helpers)
│   ├── agents/
│   │   ├── base.py           # BaseAgent class
│   │   ├── search/           # Search domain agents
│   │   │   ├── __init__.py
│   │   │   ├── verifier_agent.py
│   │   │   ├── search_agent.py
│   │   │   └── answer_agent.py
│   │   └── math/             # Math domain agents
│   │       ├── __init__.py
│   │       ├── solver_agent.py
│   │       └── verifier_agent.py
│   └── orchestra/
│       ├── base.py           # BaseOrchestra class
│       ├── search/           # Search orchestra
│       │   ├── __init__.py
│       │   └── search_orchestra.py
│       └── math/             # Math orchestra
│           ├── __init__.py
│           └── math_orchestra.py
└── multi_turn_rollout/
    └── rollout_loop.py       # Orchestra registration point
```

---

## Developing Custom Agents

### Step 1: Understanding the Base Agent

All agents must inherit from [`BaseAgent`](../agent_system/agent/agents/base.py). Here's the base class structure:

```python
class BaseAgent:
    """Abstract agent. All subclasses must implement the `call` method."""

    def __init__(
        self, 
        name: str,                              # Agent name (must match registry)
        prompt: str,                            # Agent prompt template
        wg_id: str,                             # Worker group ID
        tokenizer: PreTrainedTokenizer,         # Tokenizer for text processing
        processor,                              # Optional processor (e.g., for images)
        config: Any                             # Configuration object
    ):
        self.name = name
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.wg_id = wg_id
        
        # Optional: Define start/end tags for output parsing
        self.start_tag = None
        self.end_tag = None

    def reset(self):
        """Reset agent state (called at the beginning of each episode)."""
        pass

    def build_prompt(
        self, 
        env_obs: Dict[str, Any],   # Environment observations
        team_context: List[str],    # Outputs from other agents
        step: int                   # Current environment step
    ) -> Dict[str, Any]:
        """Build the full prompt by combining env observation with agent-specific template."""
        # Default implementation formats the prompt template
        ...

    def _generate_with_llm(
        self,
        batch: DataProto,
        actor_rollout_wg,          # LLM worker group
        agent_active_mask: np.ndarray,  # Which batch items are active
        meta_info
    ) -> Tuple[DataProto, List[str]]:
        """Generate responses using the LLM. Returns batch and decoded text responses."""
        ...

    def call(
        self,
        gen_batch: DataProto,       # Input batch
        env_obs: Dict[str, Any],    # Environment observations
        team_context: List[str],    # Context from other agents
        actor_rollout_wg,           # LLM worker group
        agent_active_mask: np.ndarray,  # Active mask
        step: int                   # Environment step
    ) -> Tuple[DataProto, List[str]]:
        """Generate a response. MUST be implemented by subclasses."""
        raise NotImplementedError
```

### Step 2: Creating a New Agent

Create a new agent by:

1. **Define the prompt template** with placeholders for `{env_prompt}`, `{team_context}`, and optionally `{step}`
2. **Inherit from `BaseAgent`** and implement the `call` method
3. **Use the `@AgentRegistry.register` decorator** to register the agent

Here's a complete example:

```python
# agent_system/agent/agents/my_domain/planner_agent.py

from typing import Dict, Any, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import general_projection
import numpy as np


# Define the agent's prompt template
PLANNER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You are a "Planner Agent". Your responsibility is to analyze the task and create a step-by-step plan to accomplish the goal.

Before creating your plan, reason about the problem within <think>...</think> tags.
Then provide your final plan within <plan>...</plan> tags.
"""


@AgentRegistry.register("Planner Agent")
class PlannerAgent(BaseAgent):
    """Agent that creates step-by-step plans for task completion."""
    
    def __init__(
        self, 
        wg_id: str, 
        tokenizer: PreTrainedTokenizer, 
        processor, 
        config: Any
    ):
        super().__init__(
            name="Planner Agent",
            prompt=PLANNER_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config
        )
        # Define output parsing tags
        self.start_tag = "<plan>"
        self.end_tag = "</plan>"
    
    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask: np.ndarray,
        step: int
    ) -> Tuple[DataProto, List[str]]:
        """Generate a plan based on the current observation and team context."""
        
        # Step 1: Build the prompt by formatting the template
        obs = self.build_prompt(env_obs, team_context, step)
        
        # Step 2: Preprocess the batch (tokenization, etc.)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        
        # Step 3: Generate with LLM
        batch, text_responses = self._generate_with_llm(
            batch, 
            actor_rollout_wg, 
            agent_active_mask, 
            gen_batch.meta_info
        )
        
        # Step 4: Parse and validate outputs
        text_responses, valids = general_projection(
            text_responses,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
            check_think_tag=True,      # Require <think>...</think>
            return_tag=True,           # Include tags in output
            return_whole_response=False # Only return parsed content
        )
        
        # Step 5: Store validation results
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_responses), dtype=object)
        
        return batch, text_responses
```

### Step 3: Registering the Agent

#### 3.1 Create the `__init__.py` file

Create an `__init__.py` file in your agent module directory:

```python
# agent_system/agent/agents/my_domain/__init__.py

from .planner_agent import PlannerAgent
from .executor_agent import ExecutorAgent  # Add other agents

__all__ = [
    "PlannerAgent",
    "ExecutorAgent",
]
```

#### 3.2 Import in the Orchestra

The agent module must be imported before agents are created. This is typically done in the orchestra's `__init__` method:

```python
import importlib
importlib.import_module("agent_system.agent.agents.my_domain")
```

### Step 4: Agent Best Practices

1. **Prompt Design**
   - Always include `{env_prompt}` to receive environment context
   - Use `{team_context}` to incorporate outputs from other agents
   - Use `{step}` if step-dependent behavior is needed
   - Define clear output format with XML-like tags for parsing

2. **Output Parsing**
   - Use `general_projection()` for standard tag-based parsing
   - Use `math_projection()` for `\boxed{}` style math answers
   - Set `check_think_tag=True` if reasoning is required

3. **Action Validation**
   - Always set `is_action_valid` in `non_tensor_batch`
   - Invalid actions can receive penalties during training

4. **Custom Agent Methods**
   - Add custom methods for agent-specific logic (e.g., `get_verification_vector()`)
   - These can be called by the orchestra for routing decisions

---

## Developing Custom Orchestras

### Step 1: Understanding the Base Orchestra

The [`BaseOrchestra`](../agent_system/agent/orchestra/base.py) provides:

```python
class BaseOrchestra:
    """Abstract orchestra coordinating multiple agent instances."""

    def __init__(
        self,
        agent_ids: List[str],                    # List of agent names
        model_ids: List[str],                    # List of model paths
        agents_to_wg_mapping: Dict[str, str],    # Agent name -> Worker Group ID
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any],
        config: Any,
    ):
        # Creates agents from registry
        self.agents: Dict[str, BaseAgent] = {
            name: AgentRegistry.create(name=name, ...)
            for name in agent_ids
        }
        self.multiagent_batch_buffer = []  # Stores agent outputs
        self.memory = None                  # Optional persistent memory

    def reset(self):
        """Reset orchestra and all agents (called at episode start)."""
        ...

    def reset_buffer(self):
        """Clear the multiagent batch buffer."""
        ...

    def save_to_buffer(self, name: str, batch: DataProto):
        """Save agent output to the buffer for training."""
        ...

    def initialize_context(self, env_obs) -> Tuple[List[str], List[str], Dict]:
        """Initialize text_actions and team_context from observations."""
        ...

    def update_memory(self, text_responses: List[str]):
        """Update persistent memory with latest responses."""
        ...

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs: Dict[str, Any],  # WG ID -> Worker Group
        active_masks: np.ndarray,
        step: int
    ) -> Tuple[List[str], List[Dict]]:
        """Run the orchestra. MUST be implemented by subclasses."""
        raise NotImplementedError
```

### Step 2: Creating a New Orchestra

Create a new orchestra by inheriting from `BaseOrchestra`:

```python
# agent_system/agent/orchestra/my_domain/my_orchestra.py

from __future__ import annotations
from typing import List, Dict, Any, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
import importlib
from verl import DataProto
import numpy as np


def update_team_context(
    agent_id: str, 
    team_context: List[str], 
    text_response: List[str], 
    agent_active_mask: np.ndarray = None
) -> List[str]:
    """Update team context with agent responses."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] += f'\nThe output of "{agent_id}": {text_response[i]}\n'
    return team_context


def update_text_action(
    text_actions: List[str], 
    text_response: List[str], 
    agent_active_mask: np.ndarray = None
) -> List[str]:
    """Update text actions with latest response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)
    
    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions


class MyMultiAgentOrchestra(BaseOrchestra):
    """
    Custom multi-agent orchestra implementing [describe your pattern].
    
    Architecture:
    - Agent 1: [Role description]
    - Agent 2: [Role description]
    - Agent N: [Role description]
    
    Execution Flow:
    [Describe the execution flow]
    """
    
    # Define agent name constants
    PLANNER_AGENT = "Planner Agent"
    EXECUTOR_AGENT = "Executor Agent"
    REVIEWER_AGENT = "Reviewer Agent"
    
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        # IMPORTANT: Import agent modules before calling super().__init__
        importlib.import_module("agent_system.agent.agents.my_domain")
        
        # Initialize base class (creates agents from registry)
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        
        # Validate required agents
        if self.PLANNER_AGENT not in self.agent_ids:
            raise ValueError(f"{self.PLANNER_AGENT} is required.")
        if self.EXECUTOR_AGENT not in self.agent_ids:
            raise ValueError(f"{self.EXECUTOR_AGENT} is required.")
        
        # Load orchestra-specific configuration
        self.max_iterations = getattr(
            self.config.agent.orchestra.my_domain, 
            "max_iterations", 
            3
        )
    
    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs: Dict[str, Any],
        active_masks: np.ndarray,
        step: int
    ) -> Tuple[List[str], List[Dict]]:
        """
        Run the orchestra.
        
        Returns:
            text_actions: Final actions to send to environment
            multiagent_batch_buffer: List of agent outputs for training
        """
        # Reset buffer at the start of each run
        self.reset_buffer()
        
        # Initialize context
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        agent_active_mask = np.logical_and(
            np.ones(len(gen_batch), dtype=bool), 
            active_masks
        ).astype(bool)
        
        # ===== STEP 1: Run Planner Agent =====
        if self.PLANNER_AGENT in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.PLANNER_AGENT]
            ]
            
            batch, text_responses = self.agents[self.PLANNER_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            # Update team context with planner output
            team_context = update_team_context(
                self.PLANNER_AGENT, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            
            # Save to buffer for training
            self.save_to_buffer(self.PLANNER_AGENT, batch)
        
        # ===== STEP 2: Run Executor Agent =====
        if self.EXECUTOR_AGENT in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.EXECUTOR_AGENT]
            ]
            
            batch, text_responses = self.agents[self.EXECUTOR_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            team_context = update_team_context(
                self.EXECUTOR_AGENT, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            self.save_to_buffer(self.EXECUTOR_AGENT, batch)
            
            # Update final actions with executor output
            text_actions = update_text_action(
                text_actions, 
                text_responses, 
                agent_active_mask
            )
        
        # ===== STEP 3: Optional Reviewer Agent =====
        if self.REVIEWER_AGENT in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.REVIEWER_AGENT]
            ]
            
            batch, text_responses = self.agents[self.REVIEWER_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            team_context = update_team_context(
                self.REVIEWER_AGENT, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            self.save_to_buffer(self.REVIEWER_AGENT, batch)
        
        return text_actions, self.multiagent_batch_buffer
```

### Step 3: Registering the Orchestra

#### 3.1 Create the `__init__.py` file

```python
# agent_system/agent/orchestra/my_domain/__init__.py

from .my_orchestra import MyMultiAgentOrchestra

__all__ = ["MyMultiAgentOrchestra"]
```

#### 3.2 Register in the Rollout Loop

Add your orchestra to [`agent_system/multi_turn_rollout/rollout_loop.py`](../agent_system/multi_turn_rollout/rollout_loop.py):

```python
# In MultiAgentTrajectoryCollector.__init__()

if orchestra_type == "search":
    from agent_system.agent.orchestra.search import SearchMultiAgentOrchestra as orchestra
elif orchestra_type == "math":
    from agent_system.agent.orchestra.math import MathMultiAgentOrchestra as orchestra
elif orchestra_type == "my_domain":  # ADD YOUR ORCHESTRA HERE
    from agent_system.agent.orchestra.my_domain import MyMultiAgentOrchestra as orchestra
else:
    raise ValueError(f"Unknown orchestra_type '{orchestra_type}'.")
```

### Step 4: Orchestra Patterns

#### Pattern 1: Sequential Chain

Agents execute in a fixed sequence, each receiving outputs from previous agents:

```
Agent 1 → Agent 2 → Agent 3 → ... → Output
```

```python
def run(self, gen_batch, env_obs, actor_rollout_wgs, active_masks, step):
    self.reset_buffer()
    text_actions, team_context, env_obs = self.initialize_context(env_obs)
    agent_active_mask = active_masks.astype(bool)
    
    for agent_name in self.agent_order:
        wg = actor_rollout_wgs[self.agents_to_wg_mapping[agent_name]]
        batch, responses = self.agents[agent_name].call(
            gen_batch, env_obs, team_context, wg, agent_active_mask, step
        )
        team_context = update_team_context(agent_name, team_context, responses, agent_active_mask)
        self.save_to_buffer(agent_name, batch)
        text_actions = update_text_action(text_actions, responses, agent_active_mask)
    
    return text_actions, self.multiagent_batch_buffer
```

#### Pattern 2: Hierarchical Router

A router agent determines which execution agents should run:

```
Router Agent ─┬─ (condition A) → Agent A
              ├─ (condition B) → Agent B
              └─ (condition C) → Agent C
```

```python
def run(self, gen_batch, env_obs, actor_rollout_wgs, active_masks, step):
    self.reset_buffer()
    text_actions, team_context, env_obs = self.initialize_context(env_obs)
    agent_active_mask = active_masks.astype(bool)
    
    # Step 1: Router Agent
    wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ROUTER_AGENT]]
    batch, responses = self.agents[self.ROUTER_AGENT].call(
        gen_batch, env_obs, team_context, wg, agent_active_mask, step
    )
    team_context = update_team_context(self.ROUTER_AGENT, team_context, responses, agent_active_mask)
    self.save_to_buffer(self.ROUTER_AGENT, batch)
    
    # Get routing decisions
    route_vector = self.agents[self.ROUTER_AGENT].get_routing_decision(responses, agent_active_mask)
    
    # Step 2: Conditionally run execution agents
    agent_a_mask = np.logical_and(agent_active_mask, route_vector == "A").astype(bool)
    agent_b_mask = np.logical_and(agent_active_mask, route_vector == "B").astype(bool)
    
    if agent_a_mask.any():
        wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.AGENT_A]]
        batch, responses = self.agents[self.AGENT_A].call(
            gen_batch, env_obs, team_context, wg, agent_a_mask, step
        )
        team_context = update_team_context(self.AGENT_A, team_context, responses, agent_a_mask)
        self.save_to_buffer(self.AGENT_A, batch)
        text_actions = update_text_action(text_actions, responses, agent_a_mask)
    
    if agent_b_mask.any():
        wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.AGENT_B]]
        batch, responses = self.agents[self.AGENT_B].call(
            gen_batch, env_obs, team_context, wg, agent_b_mask, step
        )
        team_context = update_team_context(self.AGENT_B, team_context, responses, agent_b_mask)
        self.save_to_buffer(self.AGENT_B, batch)
        text_actions = update_text_action(text_actions, responses, agent_b_mask)
    
    return text_actions, self.multiagent_batch_buffer
```

#### Pattern 3: Iterative Refinement Loop

Agents iterate until a condition is met:

```
┌─→ Agent A ─→ Agent B (Validator) ─┬─ (approved) → Exit
│                                    └─ (rejected) ─┘
└────────────────────────────────────────────────────┘
```

```python
def run(self, gen_batch, env_obs, actor_rollout_wgs, active_masks, step):
    self.reset_buffer()
    text_actions, team_context, env_obs = self.initialize_context(env_obs)
    
    approved_vector = np.zeros(len(gen_batch), dtype=bool)
    
    for loop_i in range(self.max_loop_num):
        # Run Agent A on not-yet-approved items
        agent_a_mask = np.logical_and(active_masks, ~approved_vector).astype(bool)
        
        if agent_a_mask.any():
            wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.AGENT_A]]
            batch, responses = self.agents[self.AGENT_A].call(
                gen_batch, env_obs, team_context, wg, agent_a_mask, step
            )
            team_context = update_team_context(self.AGENT_A, team_context, responses, agent_a_mask)
            self.save_to_buffer(self.AGENT_A, batch)
            text_actions = update_text_action(text_actions, responses, agent_a_mask)
        
        # Skip validator on last iteration
        if loop_i == self.max_loop_num - 1:
            break
        
        # Run Validator on not-yet-approved items
        validator_mask = np.logical_and(active_masks, ~approved_vector).astype(bool)
        
        if validator_mask.any():
            wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.VALIDATOR]]
            batch, responses = self.agents[self.VALIDATOR].call(
                gen_batch, env_obs, team_context, wg, validator_mask, step
            )
            team_context = update_team_context(self.VALIDATOR, team_context, responses, validator_mask)
            self.save_to_buffer(self.VALIDATOR, batch)
            
            # Update approval status
            approved_vector = self.agents[self.VALIDATOR].update_approved_vector(
                responses, approved_vector, validator_mask
            )
        
        if approved_vector.all():
            break
    
    return text_actions, self.multiagent_batch_buffer
```

---

## Configuration Guide

### Agent Configuration

Configure your multi-agent system in the training script or YAML config:

```yaml
agent:
  multi_agent: True
  agent_ids: ["Planner Agent", "Executor Agent", "Reviewer Agent"]
  model_ids: ["Qwen/Qwen3-4B", "Qwen/Qwen3-4B", "Qwen/Qwen3-4B"]
  model_sharing: False    # False = each agent gets dedicated LLM
  orchestra_type: my_domain
  
  # Orchestra-specific settings
  orchestra:
    my_domain:
      max_iterations: 3
```

### Model Sharing

- **`model_sharing: False`** (Non-sharing / Heterogeneous)
  - Each agent has its own dedicated LLM worker group
  - Supports different models per agent
  - Higher GPU memory usage
  
- **`model_sharing: True`** (Sharing / Homogeneous)
  - Agents with the same model share an LLM worker group
  - Lower GPU memory usage
  - All agents using the same model must have identical training configs

### Per-Agent Parameter Overrides

Override training parameters per agent:

```yaml
agent:
  agent_ids: ["Agent 1", "Agent 2", "Agent 3"]
  model_ids: ["model/path", "model/path", "model/path"]
  
  agent_specific_parameters:
    # Learning rate per agent
    actor.optim.lr: [1e-6, 1e-6, 1e-7]
    # Micro batch size per agent
    actor.ppo_micro_batch_size_per_gpu: [4, 8, 8]
```

The list order corresponds to `agent_ids` order. Supported parameters include any field under `actor_rollout_ref`.

### Command Line Override Example

```bash
python3 -m verl.trainer.main_ppo \
    agent.agent_ids='["Planner Agent","Executor Agent"]' \
    agent.model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B"]' \
    agent.model_sharing=False \
    agent.orchestra_type=my_domain \
    agent.orchestra.my_domain.max_iterations=5 \
    +agent.agent_specific_parameters.actor.optim.lr='[1e-6,1e-7]' \
    # ... other configs
```

---

## Running Your Multi-Agent System

### 1. Create Training Script

Create a shell script similar to existing examples:

```bash
#!/bin/bash
# examples/my_trainer/run_my_domain.sh

set -x

##################### Agent Configurations #####################
agent_ids='["Planner Agent","Executor Agent","Reviewer Agent"]'
model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
model_sharing=False

orchestra_type=my_domain
max_iterations=3

# Agent-specific parameters
actor_optim_lr='[1e-6,1e-6,1e-7]'
actor_ppo_micro_batch_size_per_gpu='[4,4,4]'

##################### Run Training #####################
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.group_by_agent_id=True \
    data.train_files=$HOME/data/my_domain/train.parquet \
    data.val_files=$HOME/data/my_domain/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=64 \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.my_domain.max_iterations=$max_iterations \
    env.env_name=my_env \
    env.max_steps=10 \
    env.rollout.n=8 \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=2
```

### 2. Run Training

```bash
bash examples/my_trainer/run_my_domain.sh
```

### 3. Run Evaluation

```bash
bash examples/my_trainer/run_my_domain.sh eval
```

---

## Summary

To create a custom multi-agent LLM system:

1. **Create Agents**
   - Inherit from `BaseAgent`
   - Define prompt template with `{env_prompt}`, `{team_context}`, `{step}`
   - Implement `call()` method
   - Register with `@AgentRegistry.register("Agent Name")`

2. **Create Orchestra**
   - Inherit from `BaseOrchestra`
   - Import agent modules in `__init__`
   - Implement `run()` method with your execution logic
   - Register in [`rollout_loop.py`](../agent_system/multi_turn_rollout/rollout_loop.py)

3. **Configure & Run**
   - Set `agent_ids`, `model_ids`, `orchestra_type`
   - Optionally configure `model_sharing` and `agent_specific_parameters`
   - Run with training script

For detailed examples, refer to the existing implementations:
- Search: [`agent_system/agent/agents/search/`](../agent_system/agent/agents/search/) and [`agent_system/agent/orchestra/search/`](../agent_system/agent/orchestra/search/)
- Math: [`agent_system/agent/agents/math/`](../agent_system/agent/agents/math/) and [`agent_system/agent/orchestra/math/`](../agent_system/agent/orchestra/math/)
