from typing import List, Tuple
import numpy as np
import re

def general_projection(text_repsonses: List[str], start_tag: str, end_tag: str, check_think_tag: bool = False) -> List[str]:
    """
    An function to process the text_repsonses
    text_repsonses: the list of text_repsonses to be processeed, it is a list of strings.
    start_tag: the start tag to be used for projection, e.g., "<action>"
    end_tag: the end tag to be used for projection, e.g., "</action>
    """
    valids = [0] * len(text_repsonses)

    for i in range(len(text_repsonses)):
        original_str = text_repsonses[i]  # keep the original string
        start_idx = text_repsonses[i].find(start_tag)
        end_idx = text_repsonses[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                valids[i] = 0
                text_repsonses[i] = ""
                continue

            extracted_action = text_repsonses[i][start_idx + len(start_tag):end_idx]

            text_repsonses[i] = extracted_action.strip()
            valids[i] = 1

        except:
            valids[i] = 0
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