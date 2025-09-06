
from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import general_projection
import numpy as np

PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Search Agent", and your primary responsibility is to that transforms long raw search results into concise, de-noised key facts with citations.


Once you've finished your reasoning, write the final search query inside {start_tag} {end_tag} tags. Ensure your query is precise, information-rich, avoids vagueness, and maximizes the likelihood of retrieving valuable, directly relevant information rather than repeating what is already known.
"""

@AgentRegistry.register("Information Integrator")
class InformationIntegratorAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Information Integrator", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<search>"
        self.end_tag = "</search>"
    
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
        text_repsonses, valids = general_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)
        
        # team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses
    