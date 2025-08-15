from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Planning Agent", and your role within your team is to formulate a high-level plan and identify the most appropriate strategic objective.

Your responsibilities are strictly limited to:
- Formulating a high-level plan that addresses the current situation
- Ensuring the plan aligns with long-term task success

You are now at step {step}. Based on all information above, you should first reason step-by-step about the planning process. This reasoning process MUST be enclosed within <think> </think> tags.  
Once you've finished your reasoning, present your final plan enclosed within {start_tag} {end_tag} tags.
"""



@AgentRegistry.register("Planning Agent")
class PlanningAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Planning Agent", PROMPT, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<plan>"
        self.end_tag = "</plan>"
    
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context