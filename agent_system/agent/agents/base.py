from __future__ import annotations
"""
Agent definitions.
"""
from typing import Dict, Any, List, Tuple
import copy
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.protocol import extract_dataproto_via_active_mask, restore_dataproto_via_active_mask
from transformers import PreTrainedTokenizer
import numpy as np

class BaseAgent:
    """Abstract agent.  All subclasses *must* implement :py:meth:`act`."""

    def __init__(self, name: str, prompt: str, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        self.name = name
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        
        self.wg_id = wg_id

        self.start_tag = None
        self.end_tag = None

        # Check if prompt is defined for this agent via calling the property
        if not hasattr(self, 'prompt') or not isinstance(self.prompt, str):
            raise ValueError(f"Agent '{self.name}' must define a 'prompt' property.")

    def reset(self):
        pass

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str], step: int) -> str:
        """Build the prompt for the agent based on the observation."""
        # Naive Implementation
        obs = copy.deepcopy(env_obs)
        bs = len(obs['text'])
        for i in range(bs):
            if self.start_tag is not None and self.end_tag is not None:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i],
                                                    step=step,
                                                    start_tag=self.start_tag,
                                                    end_tag=self.end_tag)
            else:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i],
                                                    step=step)
        return obs


    def _generate_with_llm(self, batch: DataProto, actor_rollout_wg, agent_active_mask: np.ndarray, meta_info) -> Tuple[DataProto, List[str]]:
        """Helper: prompt → input_ids → actor_rollout_wg → decoded str."""
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        batch_input = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )

        batch_input.meta_info = meta_info
        
        # extract activate data
        batch_input_extracted = extract_dataproto_via_active_mask(batch_input, agent_active_mask)
        # pad to be divisible by dp_size
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input_extracted, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        # unpad
        batch_output_extracted = unpad_dataproto(batch_output_padded, pad_size=pad_size)
        # restorr
        batch_output = restore_dataproto_via_active_mask(batch_output_extracted, agent_active_mask)

        batch = batch.union(batch_output)
        
        text_repsonses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

        # insert model name
        batch.non_tensor_batch['wg_id'] = np.array([self.wg_id] * len(batch), dtype=object)
        batch.non_tensor_batch['agent_active_mask'] = agent_active_mask

        return batch, text_repsonses

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a response based on the observation and the batch.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            team_context (List[str]): Contextual information from the team.
            actor_rollout_wg: The LLM policy for acting.
            step: environment step
        Returns:
            Tuple[DataProto, List[str], List[str]]:
                - batch (DataProto): The processed batch after generation.
                - text_repsonses (List[str]): The generated text responses.
        """
        raise NotImplementedError

__all__ = [
    "BaseAgent",
]
