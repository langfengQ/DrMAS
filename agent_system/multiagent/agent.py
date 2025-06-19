from __future__ import annotations
"""Agent definitions & registry.

Each agent calls the LLM policy (``actor_rollout_wg``) in its
:py:meth:`act` method.  Prompts for each role live in *prompt.py*.
"""
from typing import Dict, Any, Callable, Optional, List, Tuple

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto
from transformers import PreTrainedTokenizer
from agent_system.multi_turn_rollout.utils import to_list_of_dict, torch_to_numpy, filter_group_data, preprocess_batch

from agent_system.multiagent.prompt import PROMPTS

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

        # Check if prompt is defined for this agent via calling the property
        if not hasattr(self, 'prompt') or not isinstance(self.prompt, str):
            raise ValueError(f"Agent '{self.name}' must define a 'prompt' property.")
        else:
            print(f"Agent '{self.name}' initialized with prompt: {self.prompt}")

    def reset(self):
        pass

    @property
    def prompt(self) -> str:
        """Return the prompt template for the ReflexionAgent."""
        return PROMPTS[self.name]

    def build_prompt(self, obs: Dict[str, Any]) -> str:
        """Build the prompt for the agent based on the observation."""
        # Naive Implementation
        bs = len(obs['text'])
        for i in range(bs):
            obs['text'][i] = self.prompt.format(obs=obs['text'][i])
        return obs

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

        return batch, text_repsonses

    def call(
        self,
        gen_batch: DataProto,
        obs: Dict[str, Any],
        actor_rollout_wg,
    ) -> Tuple[DataProto, List[str]]:
        """Generate a response based on the observation and the batch.
        Args:
            gen_batch (DataProto): The input batch for generation.
            obs (Dict[str, Any]): Observations from the environment.
                - 'text' (List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
            actor_rollout_wg: The LLM policy for acting.
        Returns:
            Tuple[DataProto, List[str]]: A tuple containing the output batch and the text responses.
        """
        raise NotImplementedError


# =============================================================================
# Reference agents (Memory / Planner / Action) using LLM
# =============================================================================
@AgentRegistry.register("ReflexionAgent")
class ReflexionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor,config: Any):
        super().__init__("ReflexionAgent", tokenizer=tokenizer, processor=processor,config=config)

    def call(self, gen_batch: DataProto, obs: Dict[str, Any], actor_rollout_wg) -> Tuple[DataProto, List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(obs)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)
        return batch, text_repsonses  # feed to next agent


@AgentRegistry.register("PlanningAgent")
class PlanningAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("PlanningAgent", tokenizer=tokenizer, processor=processor,config=config)
    
    def call(self, gen_batch: DataProto, obs: Dict[str, Any], actor_rollout_wg) -> Tuple[DataProto, List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(obs)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)
        return batch, text_repsonses  # feed to next agent


@AgentRegistry.register("ActionAgent")
class ActionAgent(BaseAgent):
    def __init__(self, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__("ActionAgent", tokenizer=tokenizer, processor=processor,config=config)
    
    def projection(self, text_repsonses: List[str]) -> List[str]:
        return [response.strip() for response in text_repsonses]

    def call(self, gen_batch: DataProto, obs: Dict[str, Any], actor_rollout_wg) -> Tuple[DataProto, List[str]]:
        """Generate a summary of the conversation history."""
        obs = self.build_prompt(obs)
        batch = preprocess_batch(gen_batch=gen_batch, 
                                    obs=obs, 
                                    config=self.config, 
                                    tokenizer=self.tokenizer, 
                                    processor=self.processor,
                                    )
        batch, text_repsonses = self._generate_with_llm(batch, actor_rollout_wg, gen_batch.meta_info)
        text_actions = self.projection(text_repsonses)
        return batch, text_actions  # feed to next agent


# =============================================================================
__all__ = [
    "AgentRegistry",
    "BaseAgent",
    "ReflexionAgent",
    "PlanningAgent",
    "ActionAgent",
]
