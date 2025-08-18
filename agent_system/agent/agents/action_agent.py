from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent
from agent_system.agent.utils import general_projection

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are an "Action Agent", and your role within your team is to determine the final action for the current step.

You are now at step {step}. Based on all information above, please decide on the most appropriate admissible action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, select one admissible action and MUST present it enclosed within {start_tag} {end_tag} tags.
"""

@AgentRegistry.register("Action Agent")
class ActionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Action Agent", PROMPT, tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<action>"
        self.end_tag = "</action>"
    
    def projection(self, text_repsonses: List[str]) -> List[str]:
        return [response.strip() for response in text_repsonses]

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
        text_repsonses, valids = general_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag, check_think_tag=True)
        batch.non_tensor_batch['is_action_valid'] = valids

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context
