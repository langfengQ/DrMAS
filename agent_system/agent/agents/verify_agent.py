
from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import re
import numpy as np

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

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context
    

    def _generate_with_llm(self, batch: DataProto, actor_rollout_wg, meta_info) -> Tuple[DataProto, List[str]]:
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

        # pad to be divisible by dp_size
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
        # # unpad
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

        batch = batch.union(batch_output)
        
        text_repsonses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
        text_repsonses, valids = search_projection(text_repsonses)
        batch.non_tensor_batch['is_action_valid'] = valids

        return batch, text_repsonses


def _postprocess_action(action: str) -> str:
    """Trim everything *after* the first closing `</search>` or `</answer>` tag.

    This guards against a common LLM hallucination where an action contains
    several concatenated XML‑like snippets. By hard‑cutting at the first
    relevant close tag we can safely apply non‑greedy regex below.
    """
    if "</search>" in action:
        return action.split("</search>", 1)[0] + "</search>"
    if "</answer>" in action:
        return action.split("</answer>", 1)[0] + "</answer>"
    return action


def search_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project a list of LLM *actions* into (`results`, `valids`).

    Extraction logic (order matters):
        1. Grab the **first** complete ``<search>…</search>`` block (case‑insensitive).
        2. If absent, grab the **first** complete ``<answer>…</answer>`` block.
        3. If still absent, store an empty string.

    Validity logic (independent of extraction): ``valids[i]`` flips to **0** when
    the *original* action text satisfies any of:
        1. Contains **both** ``<search>`` and ``<answer>`` tags.
        2. Contains more than one ``<search>`` tag or more than one ``<answer>`` tag.

    The extracted block (if any) is **not** cleared when a validity rule fails –
    downstream callers can still inspect the fragment while trusting the flag.
    """

    results: List[str] = []
    valids: List[int] = [1] * len(actions)

    # --- Pre‑compiled patterns ------------------------------------------------
    re_search_block = re.compile(r"<search>(.*?)</search>", re.IGNORECASE | re.DOTALL)
    re_answer_block = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
    re_search_tag = re.compile(r"<search>", re.IGNORECASE)
    re_answer_tag = re.compile(r"<answer>", re.IGNORECASE)

    for i, action in enumerate(actions):
        original_action = action  # Keep untouched for validity checks
        trimmed_action = _postprocess_action(action)

        # --- Extraction -----------------------------------------------------
        m = re_search_block.search(trimmed_action)
        if m:
            results.append(f"<search>{m.group(1).strip()}</search>")
        else:
            m = re_answer_block.search(trimmed_action)
            if m:
                results.append(f"<answer>{m.group(1).strip()}</answer>")
            else:
                results.append("")
                valids[i] = 0

        # --- Validity checks -------------------------------------------------
        n_search = len(re_search_tag.findall(original_action))
        n_answer = len(re_answer_tag.findall(original_action))

        # Both search and answer present
        if n_search and n_answer:
            valids[i] = 0
            continue
        # Multiple identical tags
        if n_search > 1 or n_answer > 1:
            valids[i] = 0

    valids = np.array(valids, dtype=bool)
    
    return results, valids
