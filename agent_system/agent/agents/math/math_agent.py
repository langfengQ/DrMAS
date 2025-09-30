
from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import math_projection
import numpy as np

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{{}}.
"""

# PROMPT = (
#     r"# Task Introduction\n"
#     r"{env_prompt}\n"
#     r"# Your Teammates' Outputs at Step {step}\n"
#     r"{team_context}\n"
#     r"# Your Role\n"
#     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
#     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
#     r"The final answer MUST BE put in \boxed{{}}."
# )



@AgentRegistry.register("Math Agent")
class MathAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Math Agent", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
    
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
        text_repsonses, valids = math_projection(text_repsonses, check_think_tag=True)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)

        # team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses
