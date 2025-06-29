from typing import List

def tag_projection(text_repsonses: List[str], start_tag: str, end_tag: str) -> List[str]:
    """
    An function to process the text_repsonses
    text_repsonses: the list of text_repsonses to be processeed, it is a list of strings.
    start_tag: the start tag to be used for projection, e.g., "<action>"
    end_tag: the end tag to be used for projection, e.g., "</action>
    """
    valids = [0] * len(text_repsonses)

    for i in range(len(text_repsonses)):
        start_idx = text_repsonses[i].find(start_tag)
        end_idx = text_repsonses[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <code>...</code> block, mark as invalid
                extracted_action = text_repsonses[i][-100:]
                valids[i] = 0
                text_repsonses[i] = extracted_action
                continue

            # Extract just the content between the tags
            extracted_action = text_repsonses[i][start_idx + len(start_tag):end_idx]

            text_repsonses[i] = extracted_action
            valids[i] = 1

        except:
            extracted_action = text_repsonses[i][-100:]
            valids[i] = 0
            text_repsonses[i] = extracted_action

    return text_repsonses, valids