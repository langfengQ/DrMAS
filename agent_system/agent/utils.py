from typing import List, Tuple
import numpy as np
import re
from collections import defaultdict
from agent_system.environments.env_package.math.utils import extract_answer

from omegaconf import OmegaConf, DictConfig, ListConfig

def normalize_agent_id(agent_id: str) -> str:
    return agent_id.strip().replace(" ", "")

def normalize_model_id(model_id: str) -> str:
    return model_id.split("/")[-1]

def omega_equal_resolved(cfg1, cfg2) -> bool:
    """Check deep equality between two OmegaConf configs after resolving interpolations."""
    return OmegaConf.to_container(cfg1, resolve=True) == OmegaConf.to_container(cfg2, resolve=True)

def _set_specific_parameter(base_config, idx: int, total: int, agent_specific_parameters: List[str] = None):
    """
    Update agent-specific parameters in the base configuration for a given agent index.

    Args:
        base_config: An OmegaConf.DictConfig or a regular dict.
        idx (int): The index of the current agent (0-based).
        total (int): Total number of agents.
        agent_specific_parameters (List[str], optional):
            A list of parameter paths that vary by agent, e.g.
            ["actor.optim.lr", "actor.ppo_micro_batch_size_per_gpu"].

    Returns:
        A new configuration (DictConfig) customized for the current agent.
    """

    # Deep-copy the base config to avoid mutating the original
    cfg = OmegaConf.create(OmegaConf.to_container(base_config, resolve=True))
    if agent_specific_parameters is None:
        return cfg

    for param_path in agent_specific_parameters:
        value = OmegaConf.select(base_config, param_path)
        if value is None:
            raise ValueError(f"Parameter path '{param_path}' not found in base_config.")
        if not isinstance(value, ListConfig):
            raise ValueError(
                f"Agent-Specific Parameter '{param_path}' must be a list or ListConfig, but got {type(value).__name__}."
            )
        if len(value) != total:
            raise ValueError(
                f"Agent-Specific Parameter '{param_path}' list length mismatch: expected {total}, got {len(value)}. Full value: {value}"
            )
        
        specific_value = value[idx]
        OmegaConf.update(cfg, param_path, specific_value, merge=False)
    return cfg


def build_wg_ids(config):
    agent_ids = config.agent.agent_ids
    model_ids = config.agent.model_ids
    model_sharing = config.agent.model_sharing

    if len(agent_ids) != len(model_ids):
        raise ValueError("agent_ids and model_ids must have the same length.")


    base_config = config.actor_rollout_ref
    agent_specific_parameters = config.agent.agent_specific_parameters
    
    total_agent_num = len(agent_ids)
    if not model_sharing:
        wg_to_agents = {}
        agent_port = 0
        for idx in range(total_agent_num):
            agent_id = agent_ids[idx]
            model_id = model_ids[idx]
            per_config = _set_specific_parameter(base_config, idx, total_agent_num, agent_specific_parameters)
            per_config.rollout["agent_port"] = agent_port
            norm_agent = normalize_model_id(agent_id)
            norm_model = normalize_model_id(model_id)
            wg_id = f"{norm_model}_{norm_agent}"
            wg_to_agents[wg_id] = [{"agent_id": agent_id, "model_id": model_id, "config_actor_rollout_ref": per_config}]
            agent_port += 1
    else:
        model_to_agents = defaultdict(list)
        agent_port = 0
        for idx in range(total_agent_num):
            agent_id = agent_ids[idx]
            model_id = model_ids[idx]
            per_config = _set_specific_parameter(base_config, idx, total_agent_num, agent_specific_parameters)
            per_config.rollout["agent_port"] = agent_port
            model_to_agents[model_id].append({"agent_id": agent_id, "config_actor_rollout_ref": per_config})
            agent_port += 1

        wg_to_agents = {}
        for model_id, agents_configs in model_to_agents.items():

            ref_cfg = agents_configs[0]["config_actor_rollout_ref"]
            for ac in agents_configs[1:]:
                assert omega_equal_resolved(ref_cfg, ac["config_actor_rollout_ref"]), (
                    f"Agents sharing model '{model_id}' must have equal configs."
                )

            norm_agents = [normalize_agent_id(ac["agent_id"]) for ac in agents_configs]
            norm_model = normalize_model_id(model_id)
            wg_id = "_".join([norm_model] + norm_agents)
            wg_to_agents[wg_id] = [
                {
                    "agent_id": ac["agent_id"],
                    "model_id": model_id,
                    "config_actor_rollout_ref": ac["config_actor_rollout_ref"]
                }
                for ac in agents_configs
            ]

    return wg_to_agents

def general_projection(text_repsonses: List[str], start_tag: str, end_tag: str, check_think_tag: bool = False, return_tag: bool = False, return_whole_response: bool = False) -> List[str]:
    """
    An function to process the text_repsonses
    text_repsonses: the list of text_repsonses to be processeed, it is a list of strings.
    start_tag: the start tag to be used for projection, e.g., "<action>"
    end_tag: the end tag to be used for projection, e.g., "</action>
    check_think_tag: whether to check the <think>...</think> tag, default is False.
    return_tag: whether to return the tag, default is False.
    return_whole_response: whether to return the whole response, default is False.
    """
    valids = [0] * len(text_repsonses)

    for i in range(len(text_repsonses)):
        original_str = text_repsonses[i]  # keep the original string
        start_idx = text_repsonses[i].find(start_tag)
        end_idx = text_repsonses[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                valids[i] = 0
                if not return_whole_response:
                    text_repsonses[i] = ""
                continue
            
            extracted_action = text_repsonses[i][start_idx + len(start_tag):end_idx].strip()
            if return_whole_response:
                text_repsonses[i] = original_str[:start_idx + len(start_tag)] + extracted_action + original_str[end_idx:]
            else:
                if return_tag:
                    text_repsonses[i] = start_tag + extracted_action + end_tag
                else:
                    text_repsonses[i] = extracted_action

            valids[i] = 1

        except:
            valids[i] = 0
            if not return_whole_response:
                text_repsonses[i] = ""

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

        if check_think_tag:
            # check <think>...</think>
            think_start_idx = original_str.find("<think>")
            think_end_idx = original_str.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0

    valids = np.array(valids, dtype=bool)
    return text_repsonses, valids



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


def search_projection(actions: List[str], check_think_tag: bool = False) -> Tuple[List[str], List[int]]:
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

        if check_think_tag:
            # check <think>...</think>
            think_start_idx = original_action.find("<think>")
            think_end_idx = original_action.find("</think>")
            if think_start_idx == -1 or think_end_idx == -1:
                valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_action):
            valids[i] = 0

    valids = np.array(valids, dtype=bool)
    
    return results, valids

def math_projection(actions: List[str], check_think_tag: bool = False) -> Tuple[List[str], List[int]]:
    """
    Project a list of LLM actions into (results, valids).

    Criteria:
      1) Contains <think>...</think>;
      2) The content can be successfully extracted from the last occurrence of \\boxed{...} (according to the given logic);
    Returns:
      actions
      valids
    """
    valids: List[int] = []

    for original_str in actions:

        # 1) Check for <think>...</think>
        if check_think_tag:
            has_think = re.search(r"<think>(.*?)</think>", original_str, re.DOTALL | re.IGNORECASE) is not None
        else:
            has_think = True

        # 2) Extract using the provided boxed logic
        boxed_inner = extract_answer(original_str)

        is_valid = int(has_think and (boxed_inner is not None))
        valids.append(is_valid)

    assert len(actions) == len(valids)
    valids = np.array(valids, dtype=bool)

    return actions, valids