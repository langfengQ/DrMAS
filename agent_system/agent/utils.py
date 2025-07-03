from typing import List
import numpy as np
import re

def tag_projection(text_repsonses: List[str], start_tag: str, end_tag: str) -> List[str]:
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

    valids = np.array(valids, dtype=bool)
    return text_repsonses, valids