# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
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

The team's overall task is to: {task_description}
At each step, you and your teammates collaborate to decide the best next action.

Your team is now at "step 1" and the current observation is: {current_observation}
The admissible actions in the current situation are: [{admissible_actions}].

Now, you and your teammates must work together to determine the action.
"""

ALFWORLD_MULTIAGENT_TEMPLATE = """
You are a member of an expert multi-agent team operating in the ALFRED Embodied Environment.

The team's overall task is to: {task_description}
At each step, you and your teammates collaborate to decide the best next action. 

Prior to this step, your team has completed {step_count} environment step(s). Below is the history memory from "step 1" to "step {step_count}":
[
{memory}
]

Your team is now at "step {current_step}" and the current observation is: {current_observation}
The admissible actions in the current situation are: [{admissible_actions}].

Now, you and your teammates must work together to determine the action for step {current_step}.
"""