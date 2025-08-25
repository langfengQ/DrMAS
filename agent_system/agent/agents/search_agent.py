
from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent
from agent_system.agent.utils import search_projection
import numpy as np

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You are a "Search Agent", and your primary responsibility is to decide whether to call an external search tool or to provide a direct answer of the question.

You should first conduct a reasoning process. This process MUST be enclosed within <think> </think> tags.
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

@AgentRegistry.register("Search Agent")
class SearchAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Search Agent", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
    
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
        text_repsonses, valids = search_projection(text_repsonses, check_think_tag=False)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)

        # team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses
