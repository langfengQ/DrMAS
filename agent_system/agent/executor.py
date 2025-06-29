from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.agent import AgentRegistry, BaseAgent
from verl import DataProto


class BaseExecutor:
    """Abstract executor coordinating a list of agent *names* or instances."""

    def __init__(
        self,
        agent_names: List[str],
        tokenizer: PreTrainedTokenizer,
        processor,
        config: Any,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        if agent_names is None:
            agent_names = ["Reflexion Agent", "Action Agent", "Memory Agent"]
        
        self.agents: Dict[str, BaseAgent] = {
            name: AgentRegistry.create(name=name, 
                                       tokenizer=tokenizer,
                                       processor=processor,
                                       config=config,
                                       )
            for name in agent_names
        }
        self.agent_names = agent_names
        self.multiagent_batch_buffer: List[Dict] = []  # Buffer to store multiagent output batches
        self.memory = None

    def reset(self):
        """Reset the executor, all agents, and buffer."""
        self.reset_buffer()
        for ag in self.agents.values():
            ag.reset()

    def reset_buffer(self):
        """Clear the multiagent batch buffer before each run."""
        self.multiagent_batch_buffer.clear()

    def save_to_buffer(self, name: str, batch: DataProto):
        """Save new batch to the multiagent buffer."""
        self.multiagent_batch_buffer.append({
            "name": name,
            "batch": batch,
        })

    def initialize_context(self, env_obs):
        batch_size = len(env_obs['text'])

        team_context = ["" for _ in range(batch_size)]  # Initialize team context for each batch item
        for i in range(batch_size):
            if "{memory}" in env_obs['text'][i]:
                env_obs['text'][i] = env_obs['text'][i].replace("{memory}", self.memory[i] if self.memory is not None else "")

        return team_context, env_obs

    def update_memory(self, text_repsonses: List[str]):
        """Update the memory of the agents with the latest text responses."""
        assert "Memory Agent" in self.agent_names, "Memory Agent is required to update memory. Please add it to the agent_names list."
        self.memory = text_repsonses

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wg,
        step: int,
    ) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the executor with the given batch and environment observations.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            actor_rollout_wg: The shared LLM policy for acting.
            step: Environment step
        Returns:
            Tuple[List[str], Dict[str, DataProto]]: A tuple containing the text actions
            and a dictionary of multiagent output batches.
        """
        raise NotImplementedError


# =============================================================================
# Chain executor – sequential pass‑through
# =============================================================================
class MultiAgentChainExecutor(BaseExecutor):
    """Sequentially run agents, passing observation and batch through each agent.
    This executor runs agents in a chain, where each agent processes the output
    of the previous agent and passes its output to the next agent.
    It is useful for scenarios where agents need to work in a sequence, such as
    in a pipeline or a multi‑step process.
    Args:
        agent_names (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the executor.
    """
    def __init__(
        self,
        agent_names: Optional[List[str]] = ["Reflexion Agent", "Planning Agent", "Action Agent"],
        tokenizer: PreTrainedTokenizer = None,
        processor=None,
        config: Any = None,
    ):
        super().__init__(
            agent_names=agent_names,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
        if not self.agents:
            raise ValueError("ChainExecutor requires at least one agent.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_names
        # if self.agent_order[-1] != "ActionAgent":
        #     raise ValueError("The last agent must be ActionAgent.")

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wg, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        team_context, env_obs = self.initialize_context(env_obs)

        # run agents sequentially, passing observation and batch
        for name in self.agent_order:
            batch, text_repsonses, team_context = self.agents[name].call(gen_batch=gen_batch, env_obs=env_obs, team_context=team_context, actor_rollout_wg=actor_rollout_wg, step=step)
            if batch is None:
                continue  # skip if the agent did not produce a batch

            # save the batch to the multiagent buffer
            self.save_to_buffer(name, batch)

            if name == "Action Agent":
                text_actions = text_repsonses
            if name == "Memory Agent":
                self.update_memory(text_repsonses)

        # if len(self.multiagent_batch_buffer) != len(self.agent_order):
        #     raise Warning("Multiagent output batch buffer length does not match number of agents. This may lead to unexpected behavior.")
        return text_actions, self.multiagent_batch_buffer


# =============================================================================
# Hierarchical executor (very simple two‑level demo)
# =============================================================================
class MultiAgentHierarchicalExecutor(BaseExecutor):
    """Example hierarchy:
        * Level‑0 (planner) → produces *sub‑goal* tokens
        * Level‑1 (action)  → acts until sub‑goal considered reached (stub)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.agents) < 2:
            raise ValueError("Hierarchy requires ≥2 agents (planner + actor).")
        self.planner = self.agents[0]
        self.actor_chain = ChainExecutor(  # reuse chain for lower level
            agent_names=[ag.name for ag in self.agents[1:]],
            tokenizer=self.tokenizer,
            config=self.config,
        )

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wg, step: int) -> str:  # noqa: D401 – override
        # self.actor_chain.reset()
        # # 1) high‑level plan ---------------------------------------------
        # plan = self.planner.call(
        #     input_text=observation,
        #     ctx={},
        #     actor_rollout_wg=actor_rollout_wg,
        #     meta_info=meta_info,
        # )
        # # 2) delegate to lower‑level chain, feeding plan into ctx ---------
        # action = self.actor_chain.run(
        #     observation=observation,
        #     actor_rollout_wg=actor_rollout_wg,
        #     meta_info=meta_info,
        # )
        # return action
        pass


__all__ = ["ChainExecutor", "HierarchicalExecutor"]
