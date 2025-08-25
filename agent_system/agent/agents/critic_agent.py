
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

# Your Teammates' Outputs at Step {step}
{team_context}

# Your Role
You are a "Critic Agent". Your job is to strictly evaluate whether the latest output of "Search Agent" at step {step} is correct, reasonable, and fully aligned with the given question. Your critique should be based on your own prior knowledge, all past search queries (<search>...</search>) and all retrieved information (<information>...</information>).

The "Search Agent" may produce one of two types of outputs: 
(1) <search>Q</search>: calling an external search engine with query Q to gather missing information.
(2) <answer>A</answer>: giving a direct answer A of the given question.

Your responsibilities:
- Actively look for any problems, weaknesses, or gaps in the latest output of "Search Agent".
- If the output is flawless (fully reasonable, precise, relevant, and leaves no room for doubt): return <critic>approve</critic>.
- If there is any issue (e.g., query is vague, irrelevant, or suboptimal; answer is imprecise, unsupported, or incomplete): return <critic>reject</critic>.

You must first provide your step-by-step reasoning enclosed in <think>...</think> tags. Once you've finished your reasoning, give your verdict using <critic>...</critic>.
"""

@AgentRegistry.register("Critic Agent")
class CriticAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Critic Agent", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<critic>"
        self.end_tag = "</critic>"

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
        text_repsonses, valids = general_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag, check_think_tag=True, return_whole_response=True)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)

        # team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses
    
    def update_approved_vector(self, text_repsonses: List[str], approved_vector: np.ndarray, agent_active_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a vector indicating whether the response is approved."""
        if agent_active_mask is None:
            agent_active_mask = np.ones(len(text_repsonses), dtype=bool)  # Default to all agents being active
        
        new_approved_vector = []
        for i in range(len(text_repsonses)):
            if agent_active_mask[i]:
                if "<critic>approve</critic>" in text_repsonses[i]:
                    new_approved_vector.append(True)
                elif "<critic>reject</critic>" in text_repsonses[i]:
                    new_approved_vector.append(False)
                else:
                    new_approved_vector.append(False)
            else:
                new_approved_vector.append(True)

        new_approved_vector = np.array(new_approved_vector, dtype=bool)
        updated_vector = np.logical_or(approved_vector, new_approved_vector).astype(bool)
        return updated_vector


