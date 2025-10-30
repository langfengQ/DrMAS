
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

# Your Role
You are a "Verifier Agent" acting as a router. Your job is to analyze the team's past search queries and reflect on their quality, efficiency, and alignment with the task goal, and then determine whether the current information and historical context are sufficient to answer the question. Based on this assessment, you will decide how to route the task.

Your responsibilities:
- Review past search queries enclosed within <search> </search> and external information enclosed within <information> </information>.
- Evaluate whether previous queries were reasonable and aligned with the task objective.
- Identify potential issues (if any), including repeated or redundant queries; imprecise queries that are too broad, vague, or missing critical constraints/entities; misaligned queries that drift away from the actual task goal.
- Assess whether the available information is complete and sufficient to generate a high-quality answer, and make a routing decision based on information sufficiency.

You are now at step {step}. You MUST first provide your detailed reasoning enclosed within <think> </think> tags. After completing your reasoning, give your routing decision:
(1) If the information is sufficient to answer the question: return <verify>yes</verify>
(2) If the information is insufficient to answer the question: return <verify>no</verify>
"""

@AgentRegistry.register("Verifier Agent")
class VerifierAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Verifier Agent", PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<verify>"
        self.end_tag = "</verify>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate verification decision."""
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        text_repsonses, valids = general_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag, check_think_tag=True, return_tag=False, return_whole_response=True)
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)

        return batch, text_repsonses
    
    def get_verification_vector(self, text_repsonses: List[str], agent_active_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a vector indicating whether information is sufficient (yes=True, no=False)."""
        if agent_active_mask is None:
            agent_active_mask = np.ones(len(text_repsonses), dtype=bool)
        
        verification_vector = []
        for i in range(len(text_repsonses)):
            if agent_active_mask[i]:
                if "<verify>yes</verify>" in text_repsonses[i].lower():
                    verification_vector.append(True)
                elif "<verify>no</verify>" in text_repsonses[i].lower():
                    verification_vector.append(False)
                else:
                    verification_vector.append(True)
            else:
                # Inactive agents are considered as needing more info
                verification_vector.append(False)

        return np.array(verification_vector, dtype=bool)

