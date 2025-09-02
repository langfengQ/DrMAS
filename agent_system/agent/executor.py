from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents import *
from verl import DataProto
import numpy as np


def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the observation dictionary with the text response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    # Naive append of the latest responses to observations
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the text actions with the latest response."""
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions

class BaseExecutor:
    """Abstract executor coordinating a list of agent *names* or instances."""

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any],
        config: Any,
    ):
        self.config = config
        self.tokenizers = tokenizers
        self.processors = processors
        self.agents_to_wg_mapping = agents_to_wg_mapping

        self.agents: Dict[str, BaseAgent] = {
            name: AgentRegistry.create(name=name,
                                       wg_id=self.agents_to_wg_mapping[name],
                                       tokenizer=tokenizers[self.agents_to_wg_mapping[name]],
                                       processor=processors[self.agents_to_wg_mapping[name]],
                                       config=config,
                                       )
            for name in agent_ids
        }
        self.agent_ids = agent_ids
        self.model_ids = model_ids
        self.multiagent_batch_buffer: List[Dict] = []  # Buffer to store multiagent output batches
        self.memory = None

    def reset(self):
        """Reset the executor, all agents, and buffer."""
        self.reset_buffer()
        self.memory = None
        for ag in self.agents.values():
            ag.reset()

    def reset_buffer(self):
        """Clear the multiagent batch buffer before each run."""
        self.multiagent_batch_buffer.clear()

    def save_to_buffer(self, name: str, batch: DataProto, agent_active_mask: Optional[np.ndarray] = None):
        """Save new batch to the multiagent buffer."""
        if agent_active_mask is None:
            agent_active_mask = np.ones(len(batch), dtype=bool)

        batch.non_tensor_batch['agent_active_mask'] = agent_active_mask
        self.multiagent_batch_buffer.append({
            "agent_id": name,
            "batch": batch,
        })

    def initialize_context(self, env_obs):
        batch_size = len(env_obs['text'])
        text_actions = ["" for _ in range(batch_size)]
        team_context = ["" for _ in range(batch_size)]  # Initialize team context for each batch item
        for i in range(batch_size):
            if "{memory}" in env_obs['text'][i]:
                env_obs['text'][i] = env_obs['text'][i].replace("{memory}", self.memory[i] if self.memory is not None else "")

        return text_actions, team_context, env_obs

    def update_memory(self, text_repsonses: List[str]):
        """Update the memory of the agents with the latest text responses."""
        if self.memory is None:
            self.memory = text_repsonses
        else:
            for i in range(len(self.memory)):
                self.memory[i] = text_repsonses[i] if len(text_repsonses[i]) > 1 else self.memory[i]

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        step: int,
    ) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the executor with the given batch and environment observations.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            actor_rollout_wgs: The LLM policies for acting.
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
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the executor.
    """
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("ChainExecutor requires at least one agent.")

        if self.config.agent.use_agent_memory and "Memory Agent" not in self.agent_ids:
            raise ValueError("Memory Agent is required to use agent memory. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        # if self.agent_order[-1] != "ActionAgent":
        #     raise ValueError("The last agent must be ActionAgent.")

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        # run agents sequentially, passing observation and batch
        for name in self.agent_order:
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[name]]
            batch, text_repsonses = self.agents[name].call(gen_batch=gen_batch, 
                                                            env_obs=env_obs, 
                                                            team_context=team_context, 
                                                            actor_rollout_wg=actor_rollout_wg, 
                                                            step=step)
            if batch is None:
                continue  # skip if the agent did not produce a batch
            
            team_context = update_team_context(name, team_context, text_repsonses)
            # save the batch to the multiagent buffer
            self.save_to_buffer(name, batch)

            if name == "Action Agent":
                text_actions = text_repsonses
            if self.config.agent.use_agent_memory and name == "Memory Agent":
                self.update_memory(text_repsonses)

        # if len(self.multiagent_batch_buffer) != len(self.agent_order):
        #     raise Warning("Multiagent output batch buffer length does not match number of agents. This may lead to unexpected behavior.")
        return text_actions, self.multiagent_batch_buffer

class SearchMultiAgentExecutor(BaseExecutor):
    """Sequentially run agents, passing observation and batch through each agent.
    This executor runs agents in a chain, where each agent processes the output
    of the previous agent and passes its output to the next agent.
    It is useful for scenarios where agents need to work in a sequence, such as
    in a pipeline or a multi‑step process.
    Args:
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the executor.
    """
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("ChainExecutor requires at least one agent.")

        if self.config.agent.use_agent_memory and "Memory Agent" not in self.agent_ids:
            raise ValueError("Memory Agent is required to use agent memory. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        self.random_dropout = self.config.agent.random_dropout
        self.random_dropout_ratio = self.config.agent.random_dropout_ratio

        self.output_agent = "Search Agent"
        self.critic_agent = "Critic Agent"
        self.enable_critic = self.critic_agent in self.agent_order
        # if self.agent_order[-1] != "ActionAgent":
        #     raise ValueError("The last agent must be ActionAgent.")
        self.max_loop_num = 2

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        if self.enable_critic:
            approved_vector = np.zeros(len(gen_batch), dtype=bool)  # Vector to track if the action is approved

        for num in range(self.max_loop_num):
            # run agents sequentially, passing observation and batch
            for name in self.agent_order:

                # skip last time for critic agent
                if num == self.max_loop_num - 1 and name == self.critic_agent:
                    break
                
                agent_active_mask = np.ones(len(gen_batch), dtype=bool)
                if self.random_dropout and name != self.output_agent:
                    agent_active_mask = np.random.binomial(1, self.random_dropout_ratio, size=len(gen_batch)).astype(bool)
                
                if self.enable_critic:
                    # AND for agent_active_mask and (not approved_vector)
                    agent_active_mask = np.logical_and(agent_active_mask, np.logical_not(approved_vector)).astype(bool)
                    
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[name]]
                batch, text_repsonses = self.agents[name].call(gen_batch=gen_batch, 
                                                                env_obs=env_obs, 
                                                                team_context=team_context, 
                                                                actor_rollout_wg=actor_rollout_wg, 
                                                                step=step)
                if batch is None:
                    continue  # skip if the agent did not produce a batch
                
                team_context = update_team_context(name, team_context, text_repsonses, agent_active_mask)
                # save the batch to the multiagent buffer
                self.save_to_buffer(name, batch, agent_active_mask)

                if name == self.critic_agent and self.enable_critic:
                    approved_vector = self.agents[self.critic_agent].update_approved_vector(text_repsonses, approved_vector, agent_active_mask)
                elif name == self.output_agent:
                    text_actions = update_text_action(text_actions, text_repsonses, agent_active_mask)

            if not self.enable_critic:
                break
            if approved_vector.all():
                break

        # if len(self.multiagent_batch_buffer) != len(self.agent_order):
        #     raise Warning("Multiagent output batch buffer length does not match number of agents. This may lead to unexpected behavior.")
        return text_actions, self.multiagent_batch_buffer


class MathMultiAgentExecutor(BaseExecutor):
    """Sequentially run agents, passing observation and batch through each agent.
    This executor runs agents in a chain, where each agent processes the output
    of the previous agent and passes its output to the next agent.
    It is useful for scenarios where agents need to work in a sequence, such as
    in a pipeline or a multi‑step process.
    Args:
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the executor.
    """
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("ChainExecutor requires at least one agent.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        # if self.agent_order[-1] != "ActionAgent":
        #     raise ValueError("The last agent must be ActionAgent.")

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        # run agents sequentially, passing observation and batch
        for name in self.agent_order:
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[name]]
            batch, text_repsonses = self.agents[name].call(gen_batch=gen_batch, 
                                                            env_obs=env_obs, 
                                                            team_context=team_context, 
                                                            actor_rollout_wg=actor_rollout_wg, 
                                                            step=step)
            if batch is None:
                continue  # skip if the agent did not produce a batch
            
            team_context = update_team_context(name, team_context, text_repsonses)
            # save the batch to the multiagent buffer
            self.save_to_buffer(name, batch)

            if name == "Math Agent":
                text_actions = text_repsonses

        # if len(self.multiagent_batch_buffer) != len(self.agent_order):
        #     raise Warning("Multiagent output batch buffer length does not match number of agents. This may lead to unexpected behavior.")
        return text_actions, self.multiagent_batch_buffer


__all__ = [
    "MultiAgentChainExecutor", 
    "MultiAgentHierarchicalExecutor", 
    "SearchMultiAgentExecutor",
    ]
