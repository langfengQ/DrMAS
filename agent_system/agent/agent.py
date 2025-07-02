from __future__ import annotations
"""Agent definitions & registry.

Each agent calls the LLM policy (``actor_rollout_wg``) in its
:py:meth:`act` method.  Prompts for each role live in *prompt.py*.
"""
from typing import Dict, Any, Callable, Optional, List, Tuple
import copy
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import preprocess_batch
from agent_system.agent.utils import tag_projection

from agent_system.agent.prompts import AGENT_PROMPTS

# Registry
class AgentRegistry:
    _REGISTRY: Dict[str, Callable[..., "BaseAgent"]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(agent_cls: Callable[..., "BaseAgent"]):
            if name in cls._REGISTRY:
                raise ValueError(f"Agent '{name}' already registered.")
            cls._REGISTRY[name] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def create(cls, name: str, **kwargs):
        if name not in cls._REGISTRY:
            raise KeyError(f"Unknown agent '{name}'. Registered: {list(cls._REGISTRY)}")
        return cls._REGISTRY[name](**kwargs)

    @classmethod
    def names(cls) -> List[str]:
        return list(cls._REGISTRY)


class BaseAgent:
    """Abstract agent.  All subclasses *must* implement :py:meth:`act`."""

    def __init__(self, name: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        self.name = name
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.start_tag = None
        self.end_tag = None

        # Check if prompt is defined for this agent via calling the property
        if not hasattr(self, 'prompt') or not isinstance(self.prompt, str):
            raise ValueError(f"Agent '{self.name}' must define a 'prompt' property.")

    def reset(self):
        pass

    @property
    def prompt(self) -> str:
        """Return the prompt template"""
        return AGENT_PROMPTS[self.name]

    def build_prompt(self, env_obs: Dict[str, Any], team_context: List[str]) -> str:
        """Build the prompt for the agent based on the observation."""
        # Naive Implementation
        obs = copy.deepcopy(env_obs)
        bs = len(obs['text'])
        for i in range(bs):
            if self.start_tag is not None and self.end_tag is not None:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i],
                                                    start_tag=self.start_tag,
                                                    end_tag=self.end_tag)
            else:
                obs['text'][i] = self.prompt.format(env_prompt=obs['text'][i],
                                                    team_context=team_context[i])
        return obs

    def postprocess_batch(self, team_context: List[str], text_response: str) -> List[str]:
        """Update the observation dictionary with the text response."""
        # Naive append of the latest responses to observations
        for i in range(len(team_context)):
            if team_context[i] == "": 
                team_context[i] = "Some of your teammates have already shared their thoughts for the current step. Their outputs are as follows:\n"
            team_context[i] = team_context[i] + f"\n{self.name}:\n{text_response[i]}\n"
        return team_context

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
        batch_output = actor_rollout_wg.generate_sequences(batch_input)

        batch = batch.union(batch_output)
        
        text_repsonses = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

        text_repsonses, valids  = tag_projection(text_repsonses, start_tag=self.start_tag, end_tag=self.end_tag)
        batch.non_tensor_batch['is_action_valid'] = valids

        return batch, text_repsonses

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        step: int,
    ) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a response based on the observation and the batch.
        Args:
            gen_batch (DataProto): The input batch for generation.
            env_obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            team_context (List[str]): Contextual information from the team.
            actor_rollout_wg: The LLM policy for acting.
            step: environment step
        Returns:
            Tuple[DataProto, List[str], List[str]]:
                - batch (DataProto): The processed batch after generation.
                - text_repsonses (List[str]): The generated text responses.
                - team_context (List[str]): Updated team context after processing.
        """
        raise NotImplementedError


# =============================================================================
# Reference agents (Memory / Planner / Action) using LLM
# =============================================================================
@AgentRegistry.register("Reflexion Agent")
class ReflexionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor,config: Any):
        super().__init__("Reflexion Agent", tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<reflexion>"
        self.end_tag = "</reflexion>"

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        if step == 0:
            return None, None, team_context
        
        obs = self.build_prompt(env_obs, team_context)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context

@AgentRegistry.register("Planning Agent")
class PlanningAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Planning Agent", tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<plan>"
        self.end_tag = "</plan>"
    
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(env_obs, team_context)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context


@AgentRegistry.register("Action Agent")
class ActionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Action Agent", tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<action>"
        self.end_tag = "</action>"
    
    def projection(self, text_repsonses: List[str]) -> List[str]:
        return [response.strip() for response in text_repsonses]

    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(env_obs, team_context)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context


@AgentRegistry.register("Memory Agent")
class MemoryAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("Memory Agent", tokenizer=tokenizer, processor=processor,config=config)
        self.start_tag = "<memory>"
        self.end_tag = "</memory>"
    
    def call(self, gen_batch: DataProto, env_obs: Dict[str, Any], team_context: List[str], actor_rollout_wg, step: int) -> Tuple[DataProto, List[str], List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(env_obs, team_context)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)

        team_context = self.postprocess_batch(team_context, text_repsonses)
        return batch, text_repsonses, team_context

# =============================================================================
__all__ = [
    "AgentRegistry",
    "BaseAgent",
    "ReflexionAgent",
    "PlanningAgent",
    "ActionAgent",
]
