import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from omegaconf import ListConfig
from PIL import Image
from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import collate_fn

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def preprocess_fn(
    item: int,
    gen_batch: DataProto,
    obs: Dict,
    config,
    tokenizer,
    processor = None,
):
    """
    Process a single observation sample, organizing environment observations (text and/or images) 
    into a format processable by the model.
    
    Parameters:
        item (int): Sample index in the batch
        gen_batch (DataProto): Batch data containing original prompts
        obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
    
    Returns:
        dict: Contains processed input data such as input_ids, attention_mask, etc.
    """

    raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
    data_source = gen_batch.non_tensor_batch['data_source'][item]

    # Get observation components
    obs_texts = obs.get('text', None)
    obs_images = obs.get('image', None)
    obs_anchors = obs.get('anchor', None)
    obs_text = obs_texts[item] if obs_texts is not None else None
    obs_image = obs_images[item] if obs_images is not None else None
    obs_anchor = obs_anchors[item] if obs_anchors is not None else None
    is_multi_modal = obs_image is not None

    _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

    # Build chat structure
    # obs_content = raw_prompt[0]['content']
    # if '<image>' in obs_content: 
    #     obs_content = obs_content.replace('<image>', '')

    # Build chat structure
    system_prompt = "You are a helpful and harmless assistant."
    for message in raw_prompt:
        if message['role'] == 'system':
            system_prompt = message['content']
        # if message['role'] == 'user':
        #     context_from_dataset = message['content']

    # if len(context_from_dataset) > 0 and obs_text is not None:
    #     obs_content = obs_text.replace('{placeholder_of_dataset_context}', context_from_dataset)
    # else:
    #     print(f"Warning: No text observation found!")

    obs_content = obs_text
    chat = np.array([
        {"content": system_prompt, "role": "system"},
        {"content": obs_content, "role": "user",}
        ])
    # Apply chat template
    prompt_with_chat_template = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )
    
    # Initialize return dict
    row_dict = {}
    
    # Process multimodal data
    if is_multi_modal:
        # Replace image placeholder with vision tokens
        raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
        row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
        image_inputs = processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
        if image_grid_thw is not None:
            merge_length = processor.image_processor.merge_size**2
            index = 0
            while '<image>' in prompt_with_chat_template:
                prompt_with_chat_template = prompt_with_chat_template.replace(
                    '<image>',
                    '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                    '<|vision_end|>',
                    1,
                )
                index += 1

            prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                            processor.image_token)

    else:
        raw_prompt = prompt_with_chat_template
    
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                        tokenizer=tokenizer,
                                                                        max_length=config.data.max_prompt_length,
                                                                        pad_token_id=tokenizer.pad_token_id,
                                                                        left_pad=True,
                                                                        truncation=config.data.truncation,)
    
    

    if is_multi_modal:
        position_ids = [
            get_rope_index(
                processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )
            ]  # (1, 3, seq_len)
    else:
        position_ids = compute_position_id_with_mask(attention_mask)
    

    raw_prompt_ids = tokenizer.encode(raw_prompt, add_special_tokens=False)
    if len(raw_prompt_ids) > config.data.max_prompt_length:
        if config.data.truncation == "left":
            raw_prompt_ids = raw_prompt_ids[-config.data.max_prompt_length :]
        elif config.data.truncation == "right":
            raw_prompt_ids = raw_prompt_ids[: config.data.max_prompt_length]
        elif config.data.truncation == "middle":
            left_half = config.data.max_prompt_length // 2
            right_half = config.data.max_prompt_length - left_half
            raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
        elif config.data.truncation == "error":
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {config.data.max_prompt_length}.")

    # Build final output dict
    row_dict.update({
        'input_ids': input_ids[0],
        'attention_mask': attention_mask[0],
        'position_ids': position_ids[0],
        'raw_prompt_ids': raw_prompt_ids,
        'anchor_obs': _obs_anchor,
        'index': item,
        'data_source': data_source
    })

    if config.data.get('return_raw_chat', False):
        row_dict['raw_prompt'] = chat.tolist()
    
    return row_dict


def preprocess_batch(
    gen_batch: DataProto, 
    obs: Dict, 
    config,
    tokenizer,
    processor=None,
) -> DataProto:
    """
    Process a batch of observation samples, converting environment observations into model-processable format.
    
    Parameters:
        gen_batch (DataProto): Batch data containing original prompts
        obs (Dict): Environment observation dictionary
            - 'text' (None or List[str]): Text observation data
            - 'image' (np.ndarray or torch.Tensor): Image observation data
            - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
    
    Returns:
        DataProto: Contains processed batch data with preserved metadata
    """
    batch_size = len(gen_batch.batch['input_ids'])
    processed_samples = []
    
    # Process each sample in parallel
    for item in range(batch_size):
        # Extract per-sample observations
        processed = preprocess_fn(
            item=item,
            gen_batch=gen_batch,
            obs=obs,
            config=config,
            tokenizer=tokenizer,
            processor=processor,
        )
        processed_samples.append(processed)
    
    # Aggregate batch data
    batch = collate_fn(processed_samples)
    
    # Create DataProto with preserved metadata
    new_batch = DataProto.from_single_dict(
        data=batch,
        meta_info=gen_batch.meta_info
    )

    return new_batch


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def flatten_lcm(values):
    if isinstance(values, ListConfig):
        values = list(values)
        arr = np.array(values, dtype=int)
    elif isinstance(values, (list, tuple, np.ndarray)):
        arr = np.array(values, dtype=int)
    elif isinstance(values, int):
        arr = np.array([values], dtype=int)
    else:
        raise ValueError(f"Unsupported type: {type(values)}")
    return int(np.lcm.reduce(arr))

def adjust_batch(config, data: DataProto, wg_id: str, mode="copy") -> DataProto:
    use_adaptive_bs = config.actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size
    ppo_mini_update_num = config.actor_rollout_ref.actor.ppo_mini_update_num

    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes

    size_divisor_ref = flatten_lcm(config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu) * world_size
    size_divisor_rollout = flatten_lcm(config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu) * world_size
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = flatten_lcm(config.actor_rollout_ref.actor.ppo_mini_batch_size) * world_size
    else:
        size_divisor_actor = flatten_lcm(config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu) * world_size

    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        adjusted_batch = data
    else:
        if mode == "delete":
            # Generate indices to remove, rather than indices to keep
            remove_indices = np.random.choice(bs, remainder, replace=False)
            # Sort remove_indices to maintain stability when deleting
            remove_indices = np.sort(remove_indices)
            
            # Create a boolean mask for elements to keep
            keep_mask = np.ones(bs, dtype=bool)
            keep_mask[remove_indices] = False

            keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
            # Apply the mask to keep elements in their original order
            tensor_data = data.batch[keep_mask_tensor]
            non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
            adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
            del data
        elif mode == "copy":
            to_add = size_divisor - remainder
            dup_indices = np.random.choice(bs, to_add, replace=False)
            dup_proto = data.select_idxs(dup_indices)

            adjusted_batch = DataProto.concat([data, dup_proto])
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    if use_adaptive_bs:
        adjusted_bs = len(adjusted_batch)
        ulysses_sequence_parallel_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
        assert config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) == config.critic.get("ulysses_sequence_parallel_size", 1)
        # assert adjusted_bs % ppo_mini_update_num == 0, f"Adjusted batch size {adjusted_bs} is not divisible by (update_num*node_num//ulysses_sequence_parallel_size) {ppo_mini_update_num*world_size//ulysses_sequence_parallel_size}."
        adjusted_batch.meta_info[f"{wg_id}/ppo_mini_batch_size"] = -(-adjusted_bs // (ppo_mini_update_num*world_size//ulysses_sequence_parallel_size)) # ceil division
        assert adjusted_batch.meta_info[f"{wg_id}/ppo_mini_batch_size"] > 0, "ppo_mini_batch_size must be greater than 0."

    return adjusted_batch

def split_batch_by_wg_ids(data: DataProto, unique_wg_ids: List[str], update_agent_ids: List[str] = None) -> Dict[str, DataProto]:
    """
    Split a DataProto batch into multiple batches based on unique model IDs.
    
    Parameters:
        data (DataProto): Input batch containing agent IDs in non_tensor_batch['agent_id']
        unique_wg_ids (list): List of unique workgroup IDs to split the batch by.
        update_agent_ids (List[str], optional): List of agent IDs that will be trained. If provided, all agents will be included in the output.
    Returns:
        Dict[str, DataProto]: Dictionary mapping agent IDs to their respective DataProto batches
    """
    wg_ids = data.non_tensor_batch.get('wg_id', None)
    agent_ids = data.non_tensor_batch.get('agent_id', None)
    if wg_ids is None:
        raise ValueError("DataProto does not contain 'wg_id' in non_tensor_batch.")

    split_batches = {}
    update_agent_ids = None if update_agent_ids is None else np.array(update_agent_ids, dtype=object)
    active_mask = np.ones_like(agent_ids, dtype=bool) if update_agent_ids is None \
                else np.isin(agent_ids, update_agent_ids)
        
    for _id in unique_wg_ids:
        indices = np.flatnonzero(np.logical_and((wg_ids == _id), active_mask))
        if len(indices) > 0:
            split_batches[_id] = data.select_idxs(indices)
    
    return split_batches

def combine_batches(split_batches: Dict[str, DataProto]) -> DataProto:
    """
    Combine multiple DataProto batches into a single batch.
    
    Parameters:
        split_batches (Dict[str, DataProto]): Dictionary mapping agent IDs to their respective DataProto batches
    
    Returns:
        DataProto: Combined batch containing all data from the input split batches
    """
    combined_batch = None
    
    for _id, batch in split_batches.items():
        if combined_batch is None:
            combined_batch = batch
        else:
            combined_batch = DataProto.concat([combined_batch, batch])
    
    return combined_batch

def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings

