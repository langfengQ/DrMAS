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
| **Multi-Agent Orchestration** | ✅ User-defined orchestration via `BaseOrchestra`<br>✅ Sequential, hierarchical, and conditional decision flows<br>✅ Built-in Search Orchestra (Verifier → Search/Answer)<br>✅ Built-in Math Orchestra (Solver ↔ Verifier loop) |
| **Agent-Model Assignment** | ✅ Logical agents \((1, ..., K)\) mapped to physical LLM worker groups (see [`verl/trainer/config/ppo_trainer.yaml#L296`](./verl/trainer/config/ppo_trainer.yaml#L296))<br>✅ **Non-sharing**: each agent uses its own LLM (supports heterogeneous model families/sizes/checkpoints)<br>✅ **Sharing**: agents using the same model share one LLM worker group |
| **Per-Agent Configuration** | ✅ Per-agent learning rates, PPO micro-batch sizes, and other hyperparameters<br>✅ Shared-model agents undergo consistency checks to avoid conflicting configurations<br>✅ Per-agent training overrides for fine-grained control |
| **Shared Resource Pooling** | ✅ Shared GPU pool across multiple LLM worker groups for efficient hardware utilization<br>✅ SGLang for high-throughput, low-latency inference<br>✅ Gradient updates applied independently for each worker group during optimization |
| **Environments**         | ✅ Math<br>✅ Search |
| **Model Support**        | ✅ Qwen2.5<br>✅ Qwen3<br>✅ LLaMA3.2<br>and more |
| **RL Algorithms**        | ✅ Dr.MAS<br>✅ GRPO<br>🧪 GiGPO (experimental)<br>🧪 DAPO (experimental) <br>🧪 RLOO (experimental) <br>🧪 PPO (experimental) <br>and more |

# Table of Contents

- [Installation](#installation)
  - [Install veRL](#install-verl)
  - [Install Supported Environments](#install-supported-environments)
- [Run Examples](#run-examples)
  - [Search](#search)
  - [Math](#math)
- [Usage Guide](#usage-guide)
  - [Quick Start: Register Custom Agents](#quick-start-register-custom-agents)
  - [Quick Start: Create Custom Orchestras](#quick-start-create-custom-orchestras)
  - [Agent Configuration](#agent-configuration)
- [Acknowledgement](#acknowledgement)

# Installation

## Install veRL

```bash
conda create -n DrMAS python==3.12 -y
conda activate DrMAS

pip3 install -r requirements_sglang.txt
pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir

pip3 install -e .
```

## Install Supported Environments


### 1. Search
```bash
conda activate DrMAS
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset:
```bash
cd repo_root/
python examples/data_preprocess/drmas_search.py
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

### 2. Math
Prepare the dataset:
```bash
cd repo_root/
python examples/data_preprocess/drmas_math.py
```


# Run Examples

## Search

**Search (hierarchical routing)**: a 3-agent hierarchy where **Verifier** decides whether information is sufficient; it routes to **Search Agent** (generate queries) or **Answer Agent** (final response). See [`agent_system/agent/orchestra/search/README.md`](./agent_system/agent/orchestra/search/README.md).
<p align="center">
    <img src="./docs/drmas/search_multi_agent_workflow.png" alt="Multi-Agent Workflow" width="60%">
</p>


```bash
bash examples/drmas_trainer/run_search.sh # Dr.MAS
```
```bash
bash examples/grpo_trainer/run_search.sh # GRPO
```

After training completes, evaluate the multi-agent system on the full test dataset:
```bash
bash examples/drmas_trainer/run_search.sh eval # Dr.MAS
```
```bash
bash examples/grpo_trainer/run_search.sh eval # GRPO
```

## Math

**Math (iterative refinement)**: a 2-agent loop where **Solver** proposes step-by-step solutions and **Verifier** checks them; items are iterated until approved or max loops reached. See [`agent_system/agent/orchestra/math/README.md`](./agent_system/agent/orchestra/math/README.md).
<p align="center">
    <img src="./docs/drmas/math_multi_agent_workflow.png" alt="Multi-Agent Workflow" width="45%">
</p>


```bash
bash examples/drmas_trainer/run_math.sh # Dr.MAS
```
```bash
bash examples/grpo_trainer/run_math.sh # GRPO
```

After training completes, evaluate the multi-agent system on the full test dataset:
```bash
bash examples/drmas_trainer/run_math.sh eval # Dr.MAS
```
```bash
bash examples/grpo_trainer/run_math.sh eval # GRPO
```

# Usage Guide

For a **comprehensive guide** on developing custom multi-agent LLM systems, including detailed examples and best practices, see the **[Multi-Agent Development Guide](./docs/MULTI_AGENT_DEVELOPMENT_GUIDE.md)**.

The guide covers:
- Architecture overview and core components
- Step-by-step agent creation and registration
- Orchestra development patterns
- Configuration options and per-agent parameter overrides
<!-- 
## Quick Start: Register Custom Agents

1. **Create Agent Class**: Inherit from [BaseAgent](./agent_system/agent/agents/base.py#L13) and implement the `call` method:

```python
@AgentRegistry.register("Custom Agent")
class CustomAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer, processor, config):
        super().__init__("Custom Agent", MY_PROMPT, wg_id, tokenizer, processor, config)
    
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], 
             team_context: List[str], actor_rollout_wg, 
             agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        # Build prompt, generate with LLM, parse output
        ...
```

2. **Import Agent Module** in the orchestra's `__init__`:

```python
import importlib
importlib.import_module("agent_system.agent.agents.my_agent_module")
```

## Quick Start: Create Custom Orchestras

1. **Create Orchestra Class**: Inherit from [BaseOrchestra](./agent_system/agent/orchestra/base.py#L53):

```python
class MyOrchestra(BaseOrchestra):
    def __init__(self, agent_ids, model_ids, agents_to_wg_mapping, tokenizers, processors, config):
        importlib.import_module("agent_system.agent.agents.my_agents")
        super().__init__(agent_ids, model_ids, agents_to_wg_mapping, tokenizers, processors, config)
    
    def run(self, gen_batch, env_obs, actor_rollout_wgs, active_masks, step):
        # Define the execution flow of agents
        ...
```

2. **Register Orchestra** in [rollout_loop.py](./agent_system/multi_turn_rollout/rollout_loop.py#L381):

```python
if orchestra_type == "my_orchestra":
    from agent_system.agent.orchestra.my_orchestra import MyOrchestra as orchestra
```

## Agent Configuration

```yaml
agent:
  multi_agent: True
  agent_ids: ["Agent 1", "Agent 2", "Agent 3"]
  model_ids: ["model/path/1", "model/path/2", "model/path/3"]
  model_sharing: False  # False = dedicated LLM per agent
  orchestra_type: search  # "search", "math", or your custom type
  
  # Per-agent parameter overrides (order matches agent_ids)
  agent_specific_parameters:
    actor.optim.lr: [1e-6, 1e-6, 1e-7]
    actor.ppo_micro_batch_size_per_gpu: [4, 8, 8]
```

For detailed examples and advanced patterns, see the **[Multi-Agent Development Guide](./docs/MULTI_AGENT_DEVELOPMENT_GUIDE.md)**. -->

# Acknowledgement

This codebase is built upon [verl-agent](https://github.com/langfengQ/verl-agent) and [verl](https://github.com/volcengine/verl). The Search environment is adapted from [Search-R1](https://github.com/PeterGriffinJin/Search-R1) and [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym). The Math environment is adapted from [DeepScaleR](https://github.com/rllm-org/rllm) and [DAPO](https://github.com/volcengine/verl/tree/main/recipe/dapo).

We extend our gratitude to the authors and contributors of these projects for their valuable work.