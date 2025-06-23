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
            agent_names = ["ReflexionAgent", "PlanningAgent", "ActionAgent"]
        
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

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wg,
    ) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the executor with the given batch and environment observations.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            actor_rollout_wg: The shared LLM policy for acting.
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
        agent_names: Optional[List[str]] = ["ReflexionAgent", "PlanningAgent", "ActionAgent"],
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
        if self.agent_order[-1] != "ActionAgent":
            raise ValueError("The last agent must be ActionAgent.")

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wg) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        team_context = ["" for _ in range(len(env_obs['text']))]  # Initialize team context for each batch item

        # run agents sequentially, passing observation and batch
        for name in self.agent_order:
            batch, text_repsonses, team_context = self.agents[name].call(gen_batch=gen_batch, env_obs=env_obs, team_context=team_context, actor_rollout_wg=actor_rollout_wg)
            # save the batch to the multiagent buffer
            self.save_to_buffer(name, batch)

            if name == "ActionAgent":
                text_actions = text_repsonses
                break  # stop at ActionAgent
        if len(self.multiagent_batch_buffer) != len(self.agent_order):
            raise Warning("Multiagent output batch buffer length does not match number of agents. This may lead to unexpected behavior.")
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

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wg) -> str:  # noqa: D401 – override
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
