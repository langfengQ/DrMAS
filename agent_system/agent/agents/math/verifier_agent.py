from typing import Dict, Any, List, Tuple, Optional
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import general_projection
import numpy as np


VERIFIER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Verifier Agent". Your responsibility is to critically review the most recent solution provided by the "Solver Agent". Check each reasoning step, formula, and conclusion for accuracy, completeness, and logical consistency.
At the end of your output, you MUST provide your verdict within <verify> </verify> using exactly one of:
(1) <verify>approve</verify> if all steps and the final answer are correct.
(2) <verify>reject</verify> if you detect any issue.
"""


@AgentRegistry.register("Verifier Agent")
class VerifierAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Verifier Agent", VERIFIER_PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)
        self.start_tag = "<verify>"
        self.end_tag = "</verify>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, agent_active_mask, step: int) -> Tuple[DataProto, List[str]]:
        obs = self.build_prompt(env_obs, team_context, step)
        batch = preprocess_batch(
            gen_batch=gen_batch,
            obs=obs,
            config=self.config,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info)
        text_repsonses, valids = general_projection(
            text_repsonses,
            start_tag=self.start_tag,
            end_tag=self.end_tag,
            check_think_tag=False,
            return_whole_response=True,
        )
        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)
        return batch, text_repsonses

    def update_approved_vector(self, text_repsonses: List[str], approved_vector: np.ndarray, agent_active_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if agent_active_mask is None:
            agent_active_mask = np.ones(len(text_repsonses), dtype=bool)

        new_approved_vector: List[bool] = []
        for i in range(len(text_repsonses)):
            if agent_active_mask[i]:
                if "<verify>approve</verify>" in text_repsonses[i]:
                    new_approved_vector.append(True)
                elif "<verify>reject</verify>" in text_repsonses[i]:
                    new_approved_vector.append(False)
                else:
                    new_approved_vector.append(False)
            else:
                new_approved_vector.append(True)

        new_approved_vector = np.array(new_approved_vector, dtype=bool)
        updated_vector = np.logical_or(approved_vector, new_approved_vector).astype(bool)
        return updated_vector


