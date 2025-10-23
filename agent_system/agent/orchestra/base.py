from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
# from agent_system.agent.agents import *
import importlib
from verl import DataProto
import numpy as np


def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the team context with agent responses.
    
    Args:
        agent_id: Identifier of the agent that produced the response
        team_context: Current team context for each batch item
        text_response: Text responses from the agent
        agent_active_mask: Optional mask indicating which items are active
        
    Returns:
        Updated team context list
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    # Naive append of the latest responses to observations
    for i in range(len(team_context)):
        if agent_active_mask[i]:
            team_context[i] = team_context[i] + f"""\nThe output of "{agent_id}": {text_response[i]}\n"""
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update text actions with the latest agent responses.
    
    Args:
        text_actions: Current text actions for each batch item
        text_response: New text responses from the agent
        agent_active_mask: Optional mask indicating which items are active
        
    Returns:
        Updated text actions list
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)

    for i in range(len(text_actions)):
        if agent_active_mask[i]:
            text_actions[i] = text_response[i]
    return text_actions

class BaseOrchestra:
    """Abstract orchestra coordinating a list of agent instances.
    
    This base class provides common functionality for orchestrating multiple agents
    in various execution patterns (sequential, hierarchical, etc.).
    """

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer],
        processors: Dict[str, Any],
        config: Any,
    ):
        """Initialize the base orchestra.
        
        Args:
            agent_ids: List of agent identifiers
            model_ids: List of model identifiers
            agents_to_wg_mapping: Mapping from agent names to worker group IDs
            tokenizers: Dictionary of tokenizers for each worker group
            processors: Dictionary of processors for each worker group
            config: Configuration object
        """
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
        """Reset the orchestra, all agents, and buffer."""
        self.reset_buffer()
        self.memory = None
        for ag in self.agents.values():
            ag.reset()

    def reset_buffer(self):
        """Clear the multiagent batch buffer before each run."""
        self.multiagent_batch_buffer.clear()

    def save_to_buffer(self, name: str, batch: DataProto):
        """Save agent output batch to the multiagent buffer.
        
        Args:
            agent_name: Name of the agent that produced the batch
            batch: The output batch from the agent
        """
        self.multiagent_batch_buffer.append({
            "agent_id": name,
            "batch": batch,
        })

    def initialize_context(self, env_obs):
        """Initialize execution context from environment observations.
        
        Args:
            env_obs: Environment observations containing text, image, and other data
            
        Returns:
            Tuple of (text_actions, team_context, processed_env_obs)
        """
        batch_size = len(env_obs['text'])
        text_actions = ["" for _ in range(batch_size)]
        team_context = ["" for _ in range(batch_size)]  # Initialize team context for each batch item
        for i in range(batch_size):
            if "{memory}" in env_obs['text'][i]:
                memory_content = self.memory[i] if self.memory is not None else ""
                env_obs['text'][i] = env_obs['text'][i].replace("{memory}", memory_content)

        return text_actions, team_context, env_obs

    def update_memory(self, text_repsonses: List[str]):
        """Update agent memory with latest text responses.
        
        Args:
            text_responses: Latest text responses from agents
        """
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
        active_masks,
        step: int,
    ) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the orchestra with the given batch and environment observations.
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
