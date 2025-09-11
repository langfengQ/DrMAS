from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
# from agent_system.agent.agents import *
import importlib
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


class SearchMultiAgentOrchestra(BaseOrchestra):
    """Sequentially run agents, passing observation and batch through each agent.
    This orchestra runs agents in a chain, where each agent processes the output
    of the previous agent and passes its output to the next agent.
    It is useful for scenarios where agents need to work in a sequence, such as
    in a pipeline or a multi‑step process.
    Args:
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the orchestra.
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
        importlib.import_module("agent_system.agent.agents.search")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("Orchestra requires at least one agent.")

        if self.config.agent.use_agent_memory and "Memory Agent" not in self.agent_ids:
            raise ValueError("Memory Agent is required to use agent memory. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        self.random_dropout = self.config.agent.random_dropout
        self.random_dropout_ratio = self.config.agent.random_dropout_ratio

        self.output_agent = "Search Agent"
        self.critic_agent = "Critic Agent"
        self.reflexion_agent = "Reflexion Agent"
        self.enable_critic = self.critic_agent in self.agent_order
        # if self.agent_order[-1] != "ActionAgent":
        #     raise ValueError("The last agent must be ActionAgent.")
        self.max_loop_num = 2

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)

        if self.enable_critic:
            approved_vector = np.zeros(len(gen_batch), dtype=bool)  # Vector to track if the action is approved

        for loop_i in range(self.max_loop_num):
            # run agents sequentially, passing observation and batch
            for name in self.agent_order:

                if name == self.reflexion_agent:
                    if step == 1 or loop_i != 0:
                        continue

                # skip last time for critic agent
                if name == self.critic_agent and loop_i == self.max_loop_num - 1:
                    break
                    
                agent_active_mask = np.ones(len(gen_batch), dtype=bool)
                if self.random_dropout and name != self.output_agent:
                    agent_active_mask = np.random.binomial(1, self.random_dropout_ratio, size=len(gen_batch)).astype(bool)

                agent_active_mask = np.logical_and(agent_active_mask, active_masks).astype(bool)
                
                if self.enable_critic:
                    # AND for agent_active_mask and (not approved_vector)
                    agent_active_mask = np.logical_and(agent_active_mask, np.logical_not(approved_vector)).astype(bool)
                    
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[name]]
                batch, text_repsonses = self.agents[name].call(gen_batch=gen_batch, 
                                                                env_obs=env_obs, 
                                                                team_context=team_context, 
                                                                actor_rollout_wg=actor_rollout_wg,
                                                                agent_active_mask=agent_active_mask, 
                                                                step=step,
                                                                )
                
                team_context = update_team_context(name, team_context, text_repsonses, agent_active_mask)
                # save the batch to the multiagent buffer
                self.save_to_buffer(name, batch)

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