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
    """
    The architecture consists of:
    1. Search Agent: Generates search queries to gather information
    2. Verifier Agent: Determines if information is sufficient to answer
    3. Answer Agent: Generates final answer when information is sufficient
    

   Verifier Agent (Router) → Evaluates if historical information is sufficient
        ├─ If "no" → 2a. Search Agent → Generates search query → Return search query
        └─ If "yes" → 2b. Answer Agent → Generates answer → Return answer

    Args:
        agent_ids (List[str]): List of agent names to be executed in sequence.
        tokenizer (PreTrainedTokenizer): Tokenizer for processing text.
        processor: Processor for handling data.
        config (Any): Configuration object containing settings for the orchestra.
    """
    # Agent type constants
    VERIFIER_AGENT = "Verifier Agent"
    SEARCH_AGENT = "Search Agent"
    ANSWER_AGENT = "Answer Agent"
    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        """Initialize the search multi-agent orchestra.
        
        Args:
            agent_ids: List of agent names to be executed in sequence
            model_ids: List of model identifiers
            agents_to_wg_mapping: Mapping from agent names to worker group IDs
            tokenizers: Dictionary of tokenizers for each worker group
            processors: Dictionary of processors for each worker group
            config: Configuration object containing settings for the orchestra
        """
        # Import search agents module
        importlib.import_module("agent_system.agent.agents.search")

        # Initialize base class
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

        # Validate that required agents are present
        if self.SEARCH_AGENT not in self.agent_ids:
            raise ValueError("Search Agent is required. Please add it to the agent_ids.")
        if self.VERIFIER_AGENT not in self.agent_ids:
            raise ValueError("Verifier Agent is required. Please add it to the agent_ids.")
        if self.ANSWER_AGENT not in self.agent_ids:
            raise ValueError("Answer Agent is required. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the orchestra with the three-agent architecture using Verifier as router.
        
        Execution flow:
        1. Verifier Agent determines if current information is sufficient (routing decision)
        2. If sufficient (yes): Answer Agent generates final answer
        3. If insufficient (no): Search Agent generates search query
        4. Return: Answer Agent output if verifier says yes, Search Agent output if verifier says no
        """
        # clear and reset multiagent batch buffer
        self.reset_buffer()
        text_actions, team_context, env_obs = self.initialize_context(env_obs)
        agent_active_mask = np.logical_and(np.ones(len(gen_batch), dtype=bool), active_masks).astype(bool)

        if step < self.config.env.max_steps:
            
            # Step 1: Run Verifier Agent (Router)
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.VERIFIER_AGENT]]
            
            batch, text_repsonses = self.agents[self.VERIFIER_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=agent_active_mask,
                step=step,
            )
            
            team_context = update_team_context(self.VERIFIER_AGENT, team_context, text_repsonses, agent_active_mask)
            self.save_to_buffer(self.VERIFIER_AGENT, batch)
            
            # Get verification results (True = sufficient, False = need more info)
            verification_vector = self.agents[self.VERIFIER_AGENT].get_verification_vector(text_repsonses, agent_active_mask)
            
            # Step 2: Conditionally run Search Agent (when verification says no)
            search_active_mask = np.logical_and(agent_active_mask, np.logical_not(verification_vector)).astype(bool)
            
            if search_active_mask.any():
                actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.SEARCH_AGENT]]
                
                batch, text_repsonses = self.agents[self.SEARCH_AGENT].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=team_context,
                    actor_rollout_wg=actor_rollout_wg,
                    agent_active_mask=search_active_mask,
                    step=step,
                )
                
                team_context = update_team_context(self.SEARCH_AGENT, team_context, text_repsonses, search_active_mask)
                self.save_to_buffer(self.SEARCH_AGENT, batch)
                
                # Update text_actions with Search Agent output for samples needing more info
                text_actions = update_text_action(text_actions, text_repsonses, search_active_mask)
            
            answer_active_mask = np.logical_and(agent_active_mask, verification_vector).astype(bool)
        else:
            answer_active_mask = agent_active_mask.copy()

        # Step 3: Conditionally run Answer Agent (when verification says yes)
        if answer_active_mask.any():
            actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.ANSWER_AGENT]]
            
            batch, text_repsonses = self.agents[self.ANSWER_AGENT].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=team_context,
                actor_rollout_wg=actor_rollout_wg,
                agent_active_mask=answer_active_mask,
                step=step,
            )
            
            team_context = update_team_context(self.ANSWER_AGENT, team_context, text_repsonses, answer_active_mask)
            self.save_to_buffer(self.ANSWER_AGENT, batch)
            
            # Update text_actions with Answer Agent output for verified samples
            text_actions = update_text_action(text_actions, text_repsonses, answer_active_mask)

        return text_actions, self.multiagent_batch_buffer