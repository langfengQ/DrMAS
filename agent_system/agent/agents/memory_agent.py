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
You are a "Memory Agent", and your role within your team is to maintain a complete memory for all important history details.

Your responsibilities:
- Maintain an objective and accurate log of **important observation details** and the **team's actions**.
- Do not include internal team reasoning, planning, or discussions.
- Record one entry for each environment step using the format: "Step N: your memory for this step"
- The environment observation must be a high-level summary in your own words — do NOT copy raw observation text.
- Be sure to record meaningful and high-impact details (e.g., number, price, names, and identifiers) from the observation that could inform future decisions, or help recover from incorrect or suboptimal decisions.
- In this update, append one new entry for the current step to the existing memory buffer.
- You MUST output the full memory buffer, from step 1 to the current step, including all previous entries.

You are now at step {step}. Based on all the information above, provide a complete memory buffer enclosed within {start_tag} {end_tag} tags.
"""


@AgentRegistry.register("Memory Agent")
class MemoryAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Memory Agent", PROMPT, tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<memory>"
        self.end_tag = "</memory>"
    
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