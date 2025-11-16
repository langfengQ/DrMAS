<p align="center">
    <img src="./docs/drmas/drmas_logo.png" alt="logo" width="25%">
</p>


<p align="center">
  <!-- <a href="https://arxiv.org/abs/2505.10978">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp; -->
  <a href="https://github.com/langfengQ/DrMAS">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/DrMAS">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?style=flat-square&logo=huggingface" alt="HuggingFace Models"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/DrMAS">
    <img src="https://img.shields.io/badge/Twitter-Channel-000000?style=flat-square&logo=x" alt="X Channel"></a>
</p>

`Dr.MAS` is designed for training **Multi-Agent LLM Systems** via **Reinforcement Learning (RL)**, supporting both homogeneous (shared-LLM) and heterogeneous (non-shared, multi-LLM) agent configurations.

Unlike single-agent approaches, `Dr.MAS` supports sophisticated multi-agent setups where specialized LLM-based agents collaborate to tackle complex reasoning and decision-making tasks. The framework features **flexible agent registry**, **customizable multi-agent orchestration**, **model sharing/non-sharing (e.g., heterogeneous LLMs)**, **per-agent configuration**, and **shared resource pooling**, making it well suited for training multi-agent LLM systems with RL.

<p align="center">
    <img src="./docs/drmas/drmas_framework.png" alt="framework" width="100%">
</p>

# Quick Feature Summary

| Feature Category | Supported Capabilities|
| - | - |
| **Flexible Agent Registry** | ✅ Decorator-based agent registration (`@AgentRegistry.register`)<br>✅ Clear role specialization per agent<br>✅ Easy agent composition and management |
| **Multi-Agent Orchestration** | ✅ User-defined orchestration via `BaseOrchestra`<br>✅ Sequential, hierarchical, and conditional execution<br>✅ Built-in Search Orchestra (Verifier → Search/Answer)<br>✅ Built-in Math Orchestra (Solver ↔ Verifier loop) |
| **Agent-Model Assignment** | ✅ Flexible model-sharing or dedicated-model setups<br>✅ Heterogeneous LLMs per agent (different model families/sizes/checkpoints)<br>✅ Automatic logical mapping of agents to LLM worker groups|
| **Per-Agent Configuration** | ✅ Agent-specific actor hyperparameter<br>✅ Per-agent training overrides for fine-grained control |
| **Shared Resource Pooling** | ✅ Shared GPU pool across multiple LLM worker groups for efficient hardware utilization <br>✅ SGLang for high-throughput, low-latency inference|
| **Environments**         | ✅ Math<br>✅ Search |
| **Model Support**        | ✅ Qwen2.5<br>✅ Qwen3<br>✅ LLaMA<br>and more |
| **RL Algorithms**        | ✅ GRPO<br>✅ GiGPO<br>✅ DAPO <br>✅ RLOO <br>and more |

# Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
  - [Install veRL](#install-verl)
  - [Install Supported Environments](#install-supported-environments)
- [Run Examples](#run-examples)
  - [Math](#math)
  - [Search](#search)
- [Usage Guide](#usage-guide)
  - [How to register Custom Agents](#how-to-register-custom-agents)
  - [How to Create Custom Orchestras](#how-to-create-custom-orchestras)
  - [Agent Configuration](#agent-configuration)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)


# Key Features

## 1. Flexible Agent Registry

`Dr.MAS` provides a clean, decorator-based mechanism for defining and registering specialized agents:
- Register agents using `@AgentRegistry.register("Agent Name")` decorator
- Each agent can have specialized roles (e.g., Verifier, Search, Answer, Solver)
- Centralized registry makes it straightforward to discover and manage available agents

## 2. Multi-Agent Orchestration

`Dr.MAS` supports user-defined multi-agent orchestra, which governs the agents' execution flow.

- Create custom orchestras by inheriting from `BaseOrchestra`
- Support for sequential, hierarchical, and conditional decision flows

## 3. Agent-Model Assignment

A core assignment logic maps logical agents $(1, ..., K)$ to physical LLM worker groups, enabling flexible model sharing strategies (see [here](./verl/trainer/config/ppo_trainer.yaml#L296)):
- **Non-sharing**: Each agent uses its own LLM, supporting heterogeneous model families, sizes, and checkpoints
- **Sharing**: Agents with the same model share one LLM worker group

## 4. Per-Agent Configuration

`Dr.MAS` offers fine-grained control of each agent's training behavior:

- Per-agent learning rates, PPO micro-batch sizes, and other hyperparameters.
- Shared-model agents undergo consistency checks to avoid conflicting configurations.

## 5. Shared Resource Pooling and Scheduling

`Dr.MAS` manages GPU resources for high efficiency, allowing LLM worker groups to operate within a shared resource pool:

- Shared resource pool across multiple LLM worker groups
- SGLang for fast, parallel LLM decoding
- Gradient updates applied independently for each worker group during optimizatio

# Installation

## Install veRL

```bash
conda create -n DrMAS python==3.12 -y
conda activate DrMAS

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .
pip3 install -r requirements_sglang.txt
```

## Install Supported Environments

### 1. Math
Prepare the dataset (test.parquet contains 50 examples from MATH500, 30 examples from AIME2024, and 30 examples from AIME2025):
```bash
cd repo_root/
python examples/data_preprocess/dapo_filter.py
```


### 2. Search
```bash
conda activate DrMAS
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
For fast validation during the training, sample 30 entries from each data source (total 210 samples):
```bash
cd repo_root/

# For fast validation during training (sample 30 entries per data source, total 210 samples):
python examples/data_preprocess/preprocess_search_r1_dataset.py --samples_per_source 30

# Or, to process the full test dataset:
# python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

# Run Examples

## Math

Train a 2-agent system (Solver, Verifier) for mathematical problem solving:

```bash
bash examples/multi_agent_trainer/run_math.sh
```

## Search

Train a 3-agent system (Verifier, Search, Answer) for information retrieval tasks:

```bash
bash examples/multi_agent_trainer/run_search.sh
```

# Usage Guide

## How to Register Custom Agents

To register a new agent:

1. **Create Agent Class**: Inherit from [BaseAgent](./agent_system/agent/agents/base.py#L13) and implement the `call` method:

```python
MY_AGENT_PROMPT = """
# Task
{env_prompt}

# Team Context
{team_context}

# Your Role
You are a "Custom Agent". Your task is to...
"""

@AgentRegistry.register("Custom Agent")
class CustomAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer, processor, config):
        ...
    
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], 
             team_context: List[str], actor_rollout_wg, 
             agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        ...
```

2. **Import Agent Module**: Ensure the agent module is imported before use (typically done in the orchestra's `__init__`):

```python
import importlib
importlib.import_module("agent_system.agent.agents.my_agent_module")
```

3. **Register in Configuration**: Add the agent to your configuration:

```yaml
agent:
  agent_ids: ["Custom Agent", "Other Agent"]
  model_ids: ["model/path/1", "model/path/2"]
```

## How to Create Multi-Agent Orchestras

To create a custom orchestra:

1. **Create New Orchestra Class**: Inherit from [BaseOrchestra](./agent_system/agent/orchestra/base.py#L53):

```python
class MyOrchestra(BaseOrchestra):
    # Define agent name constants
    AGENT_1 = "Agent 1"
    AGENT_2 = "Agent 2"
    
    def __init__(self, agent_ids, model_ids, agents_to_wg_mapping, tokenizers, processors, config):
        # Import agent modules
        import importlib
        importlib.import_module("agent_system.agent.agents.my_agents")
        
        ...
    
    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any],
            actor_rollout_wgs, active_masks: np.ndarray, 
            step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        """Define the execution flow of agents."""
        ...

```

2. **Register Orchestra**: Add it to the rollout loop [here](./agent_system/multi_turn_rollout/rollout_loop.py#L365):

```python
# In agent_system/multi_turn_rollout/rollout_loop.py
if orchestra_type == "my_orchestra":
    from agent_system.agent.orchestra.my_orchestra import MyOrchestra as orchestra
```

## Agent Configuration

Key configuration sections:
```yaml
agent:
  multi_agent: True
  agent_ids: ["Agent 1", "Agent 2", "Agent 3"]
  model_ids: ["model/path/1", "model/path/2", "model/path/3"]
  model_sharing: False  # Whether to share models across agents
  orchestra_type: search  # Orchestra type: "search", "math", or custom
  
  # Agent-specific parameter overrides
  # The list order corresponds to the order of agent_ids
  agent_specific_parameters:
    actor.optim.lr: [1e-6,1e-6,1e-7]
    actor.ppo_micro_batch_size_per_gpu: [4,8,8]
```

# Acknowledgement

This codebase is built upon [verl-agent](https://github.com/langfengQ/verl-agent) and [verl](https://github.com/volcengine/verl). The Search environment is adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym). The Math environment is adapted from [DeepScaleR](https://github.com/rllm-org/rllm) and [DAPO](https://github.com/volcengine/verl/tree/main/recipe/dapo).

We extend our gratitude to the authors and contributors of these projects for their valuable work.

