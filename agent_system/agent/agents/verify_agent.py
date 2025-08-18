
from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent
from agent_system.agent.utils import search_projection

# PROMPT = """
# # Task Introduction
# {env_prompt}

# # Your Teammates' Outputs at Step {step}
# {team_context}

# # Your Role
# You are a "Verifier", responsible for determining whether the anwser of the "Search Agent" is correct for the given question. 

# Typically, "Search Agent" output is <answer>A</answer>. Your should verify the anwser based on your own prior knowledge, all relevant past queries, and all contents provided within <information> blocks.  
# If fully supported and precise, approve by restating the same answer of "Search Agent" via <answer>A</answer>. If issues exist, you may either directly provide a corrected answer via <answer>refined answer</answer>, or call an external search engine to gather missing information using <search>your query</search>. 

# You should first conduct a reasoning process to evaluate the anwser's correctness of "Search Agent" at step {step}. This process MUST be enclosed within <think> </think> tags.
# Once you've finished your reasoning, present your response. Your response should be in one of the following forms: "<think>...</think> <answer>...</answer>" or "<think>...</think> <search>...</search>".
# """

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You are a "Verifier", responsible for determining whether the output of the "Search Agent" is reasonable or correct based on your own prior knowledge, all relevant past queries, and all contents provided within <information> blocks.

You should follow the logic below.
(1) If the latest "Search Agent" output is <search>Q</search>: Determine whether the query Q is reasonable for the given question. If the query Q is already strong, keep it via <answer>Q</answer>; otherwise refine it via <search>your refined query</search>.
(2) If the latest "Search Agent" output is <answer>A</answer>: Determine whether the anwser A is correct for the given question. If fully supported and precise, approve by restating the same answer of "Search Agent" via <answer>A</answer>. If issues exist, you may either directly provide a corrected concise answer via <answer>your refined answer</answer>, or call an external search engine to gather missing information using <search>your query</search>.

You should first conduct a reasoning process to evaluate the output's reasonableness and correctness of "Search Agent" at step {step}. This process MUST be enclosed within <think> </think> tags.
Once you've finished your reasoning, present your response based on above logic.
"""

@AgentRegistry.register("Verify Agent")
class VerifyAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Verify Agent", PROMPT, tokenizer=tokenizer, processor=processor, config=config)
    
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

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context

