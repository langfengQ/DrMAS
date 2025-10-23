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
        # Agent type constants
    OUTPUT_AGENT = "Search Agent"
    CRITIC_AGENT = "Critic Agent"
    REFLEXION_AGENT = "Reflexion Agent"
    MEMORY_AGENT = "Memory Agent"
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

        if self.config.agent.use_agent_memory and "Memory Agent" not in self.agent_ids:
            raise ValueError("Memory Agent is required to use agent memory. Please add it to the agent_ids.")
        
        # The order of agents is the execution order.
        self.agent_order = self.agent_ids
        self.random_dropout_ratio = self.config.agent.random_dropout_ratio

        self.enable_critic = self.CRITIC_AGENT in self.agent_order

        # Loop configuration
        self.max_loop_num = 2

        # Execution state
        self._last_text_responses: List[str] = []


    def _initialize_execution_context(self, env_obs: Dict[str, Any]) -> None:
        """Initialize the execution context."""
        self.reset_buffer()
        self.text_actions, self.team_context, self.env_obs = self.initialize_context(env_obs)

    def _setup_critic_tracking(self, gen_batch: DataProto) -> Optional[np.ndarray]:
        """Setup critic approval tracking if critic is enabled."""
        if self.enable_critic:
            return np.zeros(len(gen_batch), dtype=bool)
        return None

    def _execute_agent_loop(
        self, 
        gen_batch: DataProto, 
        env_obs: Dict[str, Any], 
        actor_rollout_wgs, 
        active_masks: np.ndarray, 
        step: int, 
        loop_i: int, 
        approved_vector: Optional[np.ndarray]
    ) -> bool:
        """Execute one iteration of the agent loop.
        
        Args:
            gen_batch: The input batch for generation
            env_obs: Observations from the environment
            actor_rollout_wgs: The LLM policies for acting
            active_masks: Boolean masks indicating active agents
            step: Environment step
            loop_i: Current loop iteration
            approved_vector: Optional approval vector for critic
            
        Returns:
            True if should continue to next loop, False otherwise
        """
        for agent_name in self.agent_order:
            # Skip agent if conditions are met
            if self._should_skip_agent(agent_name, step, loop_i):
                continue
                
            # Get active mask for this agent
            agent_active_mask = self._get_agent_active_mask(gen_batch, active_masks, approved_vector, agent_name)
            
            # Execute the agent
            self._execute_single_agent(gen_batch, env_obs, actor_rollout_wgs, agent_name, agent_active_mask, step)
            
            # Update tracking based on agent type
            self._update_agent_tracking(agent_name, approved_vector, agent_active_mask)
        
        # Check if we should continue looping
        return self._should_continue_looping(approved_vector)

    def _should_skip_agent(self, agent_name: str, step: int, loop_i: int) -> bool:
        """Determine if an agent should be skipped based on conditions."""
        # Skip reflexion agent on first step or after first loop
        if agent_name == self.REFLEXION_AGENT:
            return step == 1 or loop_i != 0
        # Skip critic agent on last loop
        if agent_name == self.CRITIC_AGENT and loop_i == self.max_loop_num - 1:
            return True
        return False

    def _get_agent_active_mask(
        self, 
        gen_batch: DataProto, 
        active_masks: np.ndarray, 
        approved_vector: Optional[np.ndarray], 
        agent_name: str
    ) -> np.ndarray:
        """Get the active mask for a specific agent."""
        agent_active_mask = np.ones(len(gen_batch), dtype=bool)
        
        # Apply random dropout if enabled (except for output agent)
        if self.random_dropout_ratio > 0.0 and agent_name != self.OUTPUT_AGENT:
            agent_active_mask = np.random.binomial(1, self.random_dropout_ratio, size=len(gen_batch)).astype(bool)
        
        # Apply global active masks
        agent_active_mask = np.logical_and(agent_active_mask, active_masks).astype(bool)
        
        # Apply critic approval mask if critic is enabled
        if self.enable_critic and approved_vector is not None:
            agent_active_mask = np.logical_and(agent_active_mask, np.logical_not(approved_vector)).astype(bool)
        
        return agent_active_mask

    def _execute_single_agent(
        self, 
        gen_batch: DataProto, 
        env_obs: Dict[str, Any], 
        actor_rollout_wgs, 
        agent_name: str, 
        agent_active_mask: np.ndarray, 
        step: int
    ) -> None:
        """Execute a single agent and update context.
        
        Args:
            gen_batch: The input batch for generation
            env_obs: Observations from the environment
            actor_rollout_wgs: The LLM policies for acting
            agent_name: Name of the agent to execute
            agent_active_mask: Active mask for the agent
            step: Environment step
        """
        actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[agent_name]]
        
        batch, text_responses = self.agents[agent_name].call(
            gen_batch=gen_batch,
            env_obs=env_obs,
            team_context=self.team_context,
            actor_rollout_wg=actor_rollout_wg,
            agent_active_mask=agent_active_mask,
            step=step,
        )
        
        # Update team context with agent response
        self.team_context = update_team_context(agent_name, self.team_context, text_responses, agent_active_mask)
        
        # Save batch to buffer
        self.save_to_buffer(agent_name, batch)
        
        # Store responses for tracking
        self._last_text_responses = text_responses

    def _update_agent_tracking(
        self, 
        agent_name: str, 
        approved_vector: Optional[np.ndarray], 
        agent_active_mask: np.ndarray
    ) -> None:
        """Update tracking based on agent type."""
        if agent_name == self.CRITIC_AGENT and self.enable_critic:
            approved_vector = self.agents[self.CRITIC_AGENT].update_approved_vector(self._last_text_responses, approved_vector, agent_active_mask)
        elif agent_name == self.OUTPUT_AGENT:
            self.text_actions = update_text_action(self.text_actions, self._last_text_responses, agent_active_mask)

    def _should_continue_looping(self, approved_vector: Optional[np.ndarray]) -> bool:
        """Determine if the orchestration should continue looping."""
        if not self.enable_critic:
            return False
        if self.enable_critic and approved_vector is not None and approved_vector.all():
            return False
        return True

    def run(
        self, 
        gen_batch: DataProto, 
        env_obs: Dict[str, Any], 
        actor_rollout_wgs, 
        active_masks: np.ndarray, 
        step: int
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Run the orchestra with the given batch and environment observations.
        
        Args:
            gen_batch: The input batch for generation
            env_obs: Observations from the environment
            actor_rollout_wgs: The LLM policies for acting
            active_masks: Boolean masks indicating active agents
            step: Environment step
            
        Returns:
            Tuple containing text actions and multiagent output batches
        """
        # Initialize execution context
        self._initialize_execution_context(env_obs)
        
        # Setup critic approval tracking if enabled
        approved_vector = self._setup_critic_tracking(gen_batch)
        
        # Execute the main orchestration loop
        for loop_i in range(self.max_loop_num):
            should_continue = self._execute_agent_loop(
                gen_batch, env_obs, actor_rollout_wgs, active_masks, 
                step, loop_i, approved_vector
            )
            
            if not should_continue:
                break
        
        return self.text_actions, self.multiagent_batch_buffer