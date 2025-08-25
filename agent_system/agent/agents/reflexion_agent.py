from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent
from agent_system.agent.utils import general_projection
import numpy as np

PROMPT = """
# Task Introduction
{env_prompt}

{team_context}

# Your Role
You are a "Reflexion Agent". Your role is to analyze the team's past search queries and point out mistakes, inefficiencies, missed opportunities, or false assumptions. Your reflection will help the your team understand what could have been done better and how to improve in future steps.

Your responsibilities:
- Review past search queries and external information (<search>...</search>, <information>...</information>).  
- Identify any wrong queries, irrelevant focus, redundant or overly broad searches.
- Suggest specific improvements for the next steps.

You are now at step {step}. You should first reason step-by-step about the past events. This reasoning process MUST be enclosed within <think> </think> tags.  
Once you've finished your reasoning, provide your final reflection enclosed within {start_tag} {end_tag} tags.
"""

@AgentRegistry.register("Reflexion Agent")
class ReflexionAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor,config: Any):
        super().__init__("Reflexion Agent", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<reflexion>"
        self.end_tag = "</reflexion>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        if step == 1:
            return None, None
        
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)
        text_repsonses, valids = general_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag, check_think_tag=False)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)

        # team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses