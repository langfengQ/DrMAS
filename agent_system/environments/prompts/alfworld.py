# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observaitons and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""


ALFWORLD_MULTIAGENT_TEMPLATE_NO_HIS = """
You are a member of an expert multi-agent team operating in the ALFRED Embodied Environment.
At each step, you and your teammates collaboratively decide on the next action.
The current observation is: {current_observation}
The admissible actions in the current situation are: [{admissible_actions}].

Now, you and your teammates must work together to determine the next action.
"""

ALFWORLD_MULTIAGENT_TEMPLATE = """
You are a member of an expert multi-agent team operating in the ALFRED Embodied Environment. The team's overall task is: {task_description}
At each step, you and your teammates collaboratively decide on the next action.

So far, your team has completed {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions taken by your team: {action_history}
Your team is now at step {current_step}, and the current observation is: {current_observation}
The admissible actions in the current situation are: [{admissible_actions}].

Now, you and your teammates must work together to determine the next action.
"""