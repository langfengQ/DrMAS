# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Any, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.utils import math_projection
import numpy as np


REFINER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Refiner Agent". Your job is to produce a corrected, complete solution to the math problem above by incorporating the "Critic Agent"'s feedback into the most recent solution from the "Solver Agent". If the critique identified a genuine error, fix it; if the critique was mistaken, you may keep the original reasoning but should still double check it. Write out the full, self-contained reasoning for your revised solution (do not just describe the fix).

You should give the final answer within \\boxed{{}}.
"""


@AgentRegistry.register("Refiner Agent")
class RefinerAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Refiner Agent", REFINER_PROMPT, wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config)

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
        text_repsonses, valids = math_projection(text_repsonses, check_think_tag=False)

        batch.non_tensor_batch['is_action_valid'] = valids
        batch.non_tensor_batch['env_step'] = np.array([step] * len(text_repsonses), dtype=object)
        return batch, text_repsonses
