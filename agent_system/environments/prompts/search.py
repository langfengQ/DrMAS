# SEARCH_TEMPLATE_NO_HIS = """
# Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {task_description}
# """

# SEARCH_TEMPLATE = """
# Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {task_description}
# {memory_context}
# """


SEARCH_TEMPLATE_NO_HIS = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_TEMPLATE = """
You are an expert agent tasked with answering the given question step-by-step.
Your question: {task_description}

Prior to this step, you have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped your past search queries and <information> </information> wrapped the corresponding search results returned by the external search engine. History:
{memory_context}

Now it's your turn to respond for the current step.
You should first conduct reasoning process. This process MUST be enclosed within <think> </think> tags. 
After completing your reasoning, choose only one of the following actions (do not perform both):
(1) If you find you lack some knowledge, you can call a search engine to get more external information using format: <search> your query </search>.
(2) If you have enough knowledge to answer the question confidently, provide your final answer within <answer> </answer> tags, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

SEARCH_MULTIAGENT_TEMPLATE_NO_HIS = """
You are a member of an expert multi-agent team tasked with answering the given question step-by-step.
The question is: {task_description}
Your team can access an external search engine to retrieve external information. At each step, you and your teammates must collaborate to make progress toward answering the question.
"""

SEARCH_MULTIAGENT_TEMPLATE = """
You are a member of an expert multi-agent team tasked with answering the given question step-by-step.
The question is: {task_description}
Your team can access an external search engine to retrieve external information. At each step, you and your teammates must collaborate to make progress toward answering the question.

Prior to this step, your team have already taken {step_count} step(s). Below is the interaction history where <search> </search> wrapped the past search queries and <information> </information> wrapped the corresponding retrieved information returned by the external search engine. History:
{memory_context}
"""



