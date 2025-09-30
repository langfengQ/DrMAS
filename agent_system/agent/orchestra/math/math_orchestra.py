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

class MathMultiAgentOrchestra(BaseOrchestra):
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
        importlib.import_module("agent_system.agent.agents.math")
        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if not self.agents:
            raise ValueError("MathMultiAgentOrchestra requires at least one agent.")
        
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