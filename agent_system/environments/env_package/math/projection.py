from typing import List, Tuple
import re

def math_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Project a list of LLM actions into (`results`, `valids`)
    """
    valids: List[int] = [1] * len(actions)

    return actions, valids
