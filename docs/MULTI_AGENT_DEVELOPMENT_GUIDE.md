# Multi-Agent LLM Systems Development Guide

This guide provides comprehensive instructions for developing custom multi-agent LLM systems using the Dr.MAS framework. It covers agent registration, orchestration, configuration, and integration with the training pipeline.

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
- [Configuration Guide](#configuration-guide)
  - [Agent Configuration](#agent-configuration)
  - [Model Sharing](#model-sharing)
  - [Per-Agent Parameter Overrides](#per-agent-parameter-overrides)
- [Running Your Multi-Agent System](#running-your-multi-agent-system)
- [Summary](#summary)

---

## Architecture Overview

Dr.MAS follows a modular architecture for multi-agent LLM systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Training Pipeline                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              MultiAgentTrajectoryCollector               │   │
│  │  ┌────────────────────────────────────────────────────┐  │   │
│  │  │                    Orchestra                       │  │   │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │  │   │
│  │  │  │ Agent 1  │  │ Agent 2  │  │ Agent N  │  ...     │  │   │
│  │  │  └──────────┘  └──────────┘  └──────────┘          │  │   │
│  │  │        │              │              │             │  │   │
│  │  │        └──────────────┼──────────────┘             │  │   │
│  │  │                       ▼                            │  │   │
│  │  │              LLM Worker Groups                     │  │   │
│  │  │       (Shared or Dedicated per Agent)              │  │   │
│  │  └────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│                        Environment                              │
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
# agent_system/agent/agents/my_domain/my_agent.py

# Define the agent's prompt template
MY_PROMPT = """
...
"""


@AgentRegistry.register("My Agent")
class MyAgent(BaseAgent):
    """Agent that creates step-by-step plans for task completion."""
    
    def __init__(
        self, 
        wg_id: str, 
        tokenizer: PreTrainedTokenizer, 
        processor, 
        config: Any
    ):
        super().__init__(
            name="My Agent",
            prompt=MY_PROMPT,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config
        )
    
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

from .my_agent import MyAgent

__all__ = [
    "MyAgent",
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
    AGENT_1 = "Agent 1"
    AGENT_2 = "Agent 2"
    AGENT_3 = "Agent 3"
    
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
        if self.AGENT_1 not in self.agent_ids:
            raise ValueError(f"{self.AGENT_1} is required.")
        if self.AGENT_2 not in self.agent_ids:
            raise ValueError(f"{self.AGENT_2} is required.")
        
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
        
        # ===== STEP 1: Run Agent 1 =====
        if self.AGENT_1 in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.AGENT_1]
            ]
            
            batch, text_responses = self.agents[self.AGENT_1].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            # Update team context with agent 1 output
            team_context = update_team_context(
                self.AGENT_1, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            
            # Save to buffer for training
            self.save_to_buffer(self.AGENT_1, batch)
        
        # ===== STEP 2: Run Agent 2 =====
        if self.AGENT_2 in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.AGENT_2]
            ]
            
            batch, text_responses = self.agents[self.AGENT_2].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            team_context = update_team_context(
                self.AGENT_2, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            self.save_to_buffer(self.AGENT_2, batch)
            
            # Update final actions with agent 2 output
            text_actions = update_text_action(
                text_actions, 
                text_responses, 
                agent_active_mask
            )
        
        # ===== STEP 3: Optional Agent 3 =====
        if self.AGENT_3 in self.agents:
            actor_rollout_wg = actor_rollout_wgs[
                self.agents_to_wg_mapping[self.AGENT_3]
            ]
            
            batch, text_responses = self.agents[self.AGENT_3].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            team_context = update_team_context(
                self.AGENT_3, 
                team_context, 
                text_responses, 
                agent_active_mask
            )
            self.save_to_buffer(self.AGENT_3, batch)
        
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

---

## Configuration Guide

### Agent Configuration

Configure your multi-agent system in the training script or YAML config:

```yaml
agent:
  multi_agent: True
  agent_ids: ["Agent 1", "Agent 2", "Agent 3"]
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
    agent.agent_ids='["Agent 1","Agent 2"]' \
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
agent_ids='["Agent 1","Agent 2","Agent 3"]'
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
