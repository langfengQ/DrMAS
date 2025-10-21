from __future__ import annotations
"""Agent execution structures – chain & hierarchy.
"""
from typing import List, Dict, Any, Optional, Tuple
from transformers import PreTrainedTokenizer
from agent_system.agent.orchestra.base import BaseOrchestra
from agent_system.agent.orchestra.performance_monitor import PerformanceMonitor
# from agent_system.agent.agents import *
import importlib
from verl import DataProto
import numpy as np


def update_team_context(agent_id: str, team_context: List[str], text_response: str, agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the observation dictionary with the text response.
    
    Optimized version: Pre-format the suffix once and apply vectorized operations.
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(team_context), dtype=bool)
    
    # Pre-compute active indices to avoid repeated conditional checks
    active_indices = np.where(agent_active_mask)[0]
    
    # Only update contexts for active samples
    for i in active_indices:
        # Use f-string formatting (more efficient than concatenation)
        team_context[i] = f'{team_context[i]}\nThe output of "{agent_id}": {text_response[i]}\n'
    
    return team_context

def update_text_action(text_actions: List[str], text_response: List[str], agent_active_mask: Optional[np.ndarray] = None) -> List[str]:
    """Update the text actions with the latest response.
    
    Optimized version: Use vectorized operations to avoid loop overhead.
    """
    if agent_active_mask is None:
        agent_active_mask = np.ones(len(text_actions), dtype=bool)
    
    # Pre-compute active indices
    active_indices = np.where(agent_active_mask)[0]
    
    # Batch update using active indices
    for i in active_indices:
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
        
        # Performance monitoring (disabled by default)
        self.perf_monitor = PerformanceMonitor(
            enabled=getattr(config.agent, 'enable_performance_monitor', False)
        )

    def run(self, gen_batch: DataProto, env_obs: Dict[str, Any], actor_rollout_wgs, active_masks: np.ndarray, step: int) -> Tuple[List[str], Dict[str, DataProto]]:
        """Run the multiagent orchestra with optimized performance.
        
        Optimizations:
        1. Early termination when no active samples remain
        2. Skip agent calls when agent_active_mask is all False
        3. Early exit from critic loop when all samples approved
        """
        with self.perf_monitor.measure("orchestra_run_total", {"batch_size": len(gen_batch), "step": step}):
            # clear and reset multiagent batch buffer
            self.reset_buffer()
            text_actions, team_context, env_obs = self.initialize_context(env_obs)

            batch_size = len(gen_batch)
        
        if self.enable_critic:
            approved_vector = np.zeros(batch_size, dtype=bool)  # Vector to track if the action is approved

        for loop_i in range(self.max_loop_num):
            # Early termination: if all samples are approved, no need to continue
            if self.enable_critic and approved_vector.all():
                break
            
            # run agents sequentially, passing observation and batch
            for name in self.agent_order:

                if name == self.reflexion_agent:
                    if step == 1 or loop_i != 0:
                        continue

                # skip last time for critic agent
                if name == self.critic_agent and loop_i == self.max_loop_num - 1:
                    break
                
                # Compute agent_active_mask efficiently
                agent_active_mask = active_masks.copy()  # Start with global active masks
                
                # Apply random dropout if enabled (but not for output agent)
                if self.random_dropout and name != self.output_agent:
                    dropout_mask = np.random.binomial(1, self.random_dropout_ratio, size=batch_size).astype(bool)
                    agent_active_mask = np.logical_and(agent_active_mask, dropout_mask)
                
                # Skip already approved samples if critic is enabled
                if self.enable_critic:
                    agent_active_mask = np.logical_and(agent_active_mask, np.logical_not(approved_vector))
                
                # Early skip: if no samples are active for this agent, skip the call entirely
                if not agent_active_mask.any():
                    continue
                
                # Measure agent execution time
                num_active = agent_active_mask.sum()
                with self.perf_monitor.measure(f"agent_{name}", {"active_samples": num_active, "loop": loop_i}):
                    actor_rollout_wg = actor_rollout_wgs[self.agents_to_wg_mapping[name]]
                    batch, text_repsonses = self.agents[name].call(gen_batch=gen_batch, 
                                                                    env_obs=env_obs, 
                                                                    team_context=team_context, 
                                                                    actor_rollout_wg=actor_rollout_wg,
                                                                    agent_active_mask=agent_active_mask, 
                                                                    step=step,
                                                                    )
                
                # Update team context only for active samples
                team_context = update_team_context(name, team_context, text_repsonses, agent_active_mask)
                
                # save the batch to the multiagent buffer
                self.save_to_buffer(name, batch)

                # Update state based on agent type
                if name == self.critic_agent and self.enable_critic:
                    approved_vector = self.agents[self.critic_agent].update_approved_vector(text_repsonses, approved_vector, agent_active_mask)
                    # Early exit if all approved after critic
                    if approved_vector.all():
                        break
                elif name == self.output_agent:
                    text_actions = update_text_action(text_actions, text_repsonses, agent_active_mask)

            # Non-critic mode: only run once
            if not self.enable_critic:
                break
        
        self.perf_monitor.record_step()
        return text_actions, self.multiagent_batch_buffer