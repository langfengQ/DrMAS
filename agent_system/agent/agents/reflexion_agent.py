from typing import Dict, Any, Callable, Optional, List, Tuple
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.registry import AgentRegistry
from agent_system.agent.base import BaseAgent

PROMPT = """
{env_prompt}

{team_context}

-------

You are a "Reflexion Agent", and your role within your team is to analyze the team's past actions and identify any mistakes, inefficiencies, missed opportunities, or incorrect assumptions that may have occurred. 
Your reflection will help the your team understand what could have been done better and how to improve in future steps.

Your responsibilities are strictly limited to:
- Review past actions, decisions, and outcomes.
- Identify mistakes, missed opportunities, inefficiencies, or false assumptions. 
- Suggest improvements that could guide better decisions in the future.

You are now at step {step}. Based on all information above, you should first reason step-by-step about the past events. This reasoning process MUST be enclosed within <think> </think> tags.  
Once you've finished your reasoning, provide a clear and insightful reflection enclosed within {start_tag} {end_tag} tags.
"""

@AgentRegistry.register("Reflexion Agent")
class ReflexionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor,config: Any):
        super().__init__("Reflexion Agent", PROMPT, tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<reflexion>"
        self.end_tag = "</reflexion>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        if step == 0:
            return None, None, team_context
        
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