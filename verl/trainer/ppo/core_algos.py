# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict, Counter

import numpy as np
import torch

import verl.utils.torch_functional as verl_F

from verl import DataProto
import uuid

from difflib import SequenceMatcher
from typing import Sequence, List, Dict, Any


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError

# ---------------------------------------------------------- #
# --------------- General Functions of GiGPO --------------- #
# ---------------------------------------------------------- #
def to_hashable(x):
    """Convert an object into a hashable type (used for clustering/grouping)."""
    if isinstance(x, (int, float, str, bool)):
        return x
    elif isinstance(x, (np.integer, np.floating)):
        return x.item()
    elif isinstance(x, np.ndarray):
        return tuple(x.flatten())
    elif isinstance(x, (list, tuple)):
        return tuple(to_hashable(e) for e in x)
    elif isinstance(x, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in x.items()))
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def summarize_group_size(group_size: list):
    """
    Summarize the dynamics of step-level group.
    Args:
        group_size : List[int]
    """
    counts = Counter(group_size)
    total = sum(counts.values())
    max_size = max(counts)

    summary = {}
    for size in range(1, max_size + 1):
        cnt = counts.get(size, 0)
        prop = cnt / total if total > 0 else 0
        summary[size] = (cnt, prop)

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size, (cnt, prop) in summary.items():
        if prop:
            print(f"{size:>4} | {cnt:>5} | {prop:>9.2%}")
            
def are_similar(a: str, b: str, threshold: float = 0.95) -> bool:
    """
    Check whether two text observations are similar enough.
    
    Args:
        a, b (str): Input strings to compare.
        threshold (float): Minimum similarity ratio.
    
    Returns:
        bool: True if similarity >= threshold.
    """
    if not isinstance(a, str) or not isinstance(b, str):
        raise ValueError("Only text-based observations are supported for similarity-based GiGPO in this version.")
    return SequenceMatcher(None, a, b).ratio() >= threshold

def compute_step_discounted_returns(batch: DataProto, gamma: float):
    """
    Compute discounted returns for each trajectory. (Eq. 5 in the paper)
    
    Args:
        batch (DataProto): Input batch.
        gamma (float): Discount factor.
    
    Returns:
        torch.Tensor: Discounted returns.
    """
    rewards = batch.non_tensor_batch['rewards'].astype(np.float32)
    traj_uids = batch.non_tensor_batch['traj_uid']
    active_masks = batch.non_tensor_batch['active_masks'].astype(np.float32)
    env_step = batch.non_tensor_batch['env_step'].astype(np.int32)
    # returns_by_traj_ = {}
    returns_by_traj = {}
    unique_traj_uids = np.unique(traj_uids)
    for uid in unique_traj_uids:
        # Get indices for this trajectory
        traj_indices = np.where(traj_uids == uid)[0]
        
        # Extract rewards and masks for this trajectory
        traj_rewards = rewards[traj_indices]
        traj_active_masks = active_masks[traj_indices]
        traj_env_step = env_step[traj_indices]
        assert traj_active_masks.all(), "active_masks should be all 1s for the same trajectory"
        
        first_of_group = np.r_[True, traj_env_step[1:] != traj_env_step[:-1]]
        step_starts = np.flatnonzero(first_of_group)
        step_ends = np.r_[step_starts[1:], len(traj_env_step)]

        step_rewards = traj_rewards[step_starts].astype(np.float32)

        # Calculate returns
        step_returns = np.zeros_like(step_rewards, dtype=np.float32)
        running_return = 0.0
        
        # Calculate returns from the end to the start
        for k in reversed(range(len(step_rewards))):
            running_return = step_rewards[k] + gamma * running_return
            step_returns[k] = running_return
        
        traj_returns = np.zeros_like(traj_rewards, dtype=np.float32)
        for sr, s, e in zip(step_returns, step_starts, step_ends):
            traj_returns[s:e] = sr
            
        # Store the results
        # returns_by_traj_[uid] = traj_returns
        returns_by_traj[uid] = (traj_indices, traj_returns)
    
    # Recombine the returns into the original batch order
    # all_returns_ = np.zeros_like(rewards)
    # for i, uid in enumerate(traj_uids):
    #     traj_indices = np.where(traj_uids == uid)[0]
    #     idx_in_traj = np.where(traj_indices == i)[0][0]  # Find position of i in its trajectory
    #     all_returns_[i] = returns_by_traj_[uid][idx_in_traj]

    all_returns = np.zeros_like(rewards, dtype=np.float32)
    for uid in unique_traj_uids:
        traj_indices, traj_returns = returns_by_traj[uid]
        all_returns[traj_indices] = traj_returns

    # assert (all_returns==all_returns_).all()
    
    all_returns = torch.tensor(all_returns, dtype=torch.float32, device=batch.batch['input_ids'].device)
    return all_returns

# ---------------------------------------------------------- #
# ---------------- Core Functions of GiGPO ----------------- #
# ---------------------------------------------------------- #

def compute_gigpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   step_rewards: torch.Tensor,
                                   response_mask: torch.Tensor,
                                   anchor_obs: np.array,
                                   index: np.array,
                                   traj_index: np.array,
                                   epsilon: float = 1e-6,
                                   step_advantage_w: float = 1.0,
                                   mode: str = "mean_norm",
                                   enable_similarity: bool = False,
                                   similarity_thresh: float = 0.95,
                                   group_by_agent_id: bool = False
                                   ):
    """
    Compute the advantages for GiGPO (https://arxiv.org/abs/2505.10978).
    """
    if mode == "mean_std_norm":
        remove_std = False
    elif mode == "mean_norm":
        remove_std = True
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Compute episode relative advantages (Eq. 3 in the paper).
    episode_advantages = episode_norm_reward(token_level_rewards, response_mask, index, traj_index, epsilon, remove_std, group_by_agent_id)
    
    # Anchor state grouping (Eq. 6 in the paper).
    step_group_uids = build_step_group(anchor_obs, index, enable_similarity, similarity_thresh)

    # Compute step relative advantages (Eq. 7 in the paper).
    step_advantages = step_norm_reward(step_rewards, response_mask, step_group_uids, epsilon, remove_std)

    # Compute joint advantages (Eq. 8 in the paper).
    scores = episode_advantages + step_advantage_w * step_advantages
    return scores, scores


def episode_norm_reward(token_level_rewards: torch.Tensor,
                        response_mask: torch.Tensor,
                        index: np.array,
                        traj_index: np.array,
                        epsilon: float = 1e-6,
                        remove_std: bool = True,
                        group_by_agent_id: bool = False,
                        ):
    """
    Compute episode-level advantage using mean-std normalization for GiGPO.
    (with only one scalar reward for each episode).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.array)`
            shape: (bs,)
        traj_index: `(np.array)`
            shape: (bs,)
        epsilon: float
            A small value to avoid division by zero.
        remove_std: bool
            If True, the standard deviation is removed from the normalization.
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard trajectory-level GRPO), the mean and std are computed across trajectories within one group.
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        episode_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask

    return episode_advantages


def build_step_group(anchor_obs: np.array, index: np.array, enable_similarity: bool = False, similarity_thresh: float = 0.95, summarize: bool = False):
    """
    Group observations by index and then cluster identical observations within each index group.
    Assigns a unique step_group_uid (UUID) to each cluster.
    
    Parameters:
    -----------
    anchor_obs : np.array
        Array of observation strings
    index : np.array
        Array of episode_group_uid
    summarize : bool
        Whether to summarize the group sizes (default: True)
    enable_similarity : bool
        Whether to enable similarity-based step-level grouping (default: False)
    similarity_thresh : float
        Threshold for similarity to consider two observations as identical (default: 1.0, meaning exact match)
    
    Returns:
    --------
    np.array
        Array of step_group_uid values corresponding to the original anchor_obs array
    """
    if enable_similarity:
        assert similarity_thresh > 0.0 and similarity_thresh < 1.0, "When enabling similarity-based step-level group, similarity_thresh should be in (0, 1)"

    # Initialize the result array with placeholder values
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    
    # Get unique indices
    unique_indices = np.unique(index)

    group_size: List[int] = []
    # Process each unique index
    for idx in unique_indices:
        if not enable_similarity:
            # Get all observations for this index using np.where
            indices = np.where(index == idx)[0]
            obs_group = anchor_obs[indices]
            
            # Create clusters for identical observations
            clusters = defaultdict(list)
            for i, obs in enumerate(obs_group):
                clusters[to_hashable(obs)].append(indices[i])  # Store the original index position
            
            # Assign unique step_group_uid to each cluster
            for obs, original_indices in clusters.items():
                # Generate a UUID for this cluster
                uid = str(uuid.uuid4())
                
                # Assign the same step_group_uid to all elements in this cluster
                group_size.append(len(original_indices))
                for original_idx in original_indices:
                    step_group_uids[original_idx] = uid
        else:
            locs = np.where(index == idx)[0]
            obs_group = anchor_obs[locs]

            # Dynamically maintain clusters: [{rep: str, locs: List[int]} ...]
            clusters: List[Dict[str, Any]] = []

            for obs, loc in zip(obs_group, locs):
                 # Try to place into an existing cluster
                placed = False
                for cluster in clusters:
                    if are_similar(obs, cluster["rep"], similarity_thresh):
                        cluster["locs"].append(loc)
                        placed = True
                        break
                # If no matching cluster, create a new one
                if not placed:
                    clusters.append({"rep": obs, "locs": [loc]})

            # Assign a UUID to each cluster
            for cluster in clusters:
                uid = str(uuid.uuid4())
                group_size.append(len(cluster["locs"]))
                for loc in cluster["locs"]:
                    step_group_uids[loc] = uid

        # Validate that all elements have been assigned a uid
    if None in step_group_uids or np.any(step_group_uids == None):
        missing_indices = np.where(step_group_uids == None)[0]
        raise ValueError(f"Failed to assign UIDs to all observations. Missing at indices: {missing_indices}")

    if summarize:
        summarize_group_size(group_size)
    print(f"Avg size of step-level group: {np.mean(group_size)}")
    return step_group_uids


def step_norm_reward(step_rewards: torch.Tensor,
                      response_mask: torch.Tensor,
                      index: np.array,
                      epsilon: float = 1e-6,
                      remove_std: bool = True,
                      ):
    """
    Compute step-level advantage using mean-std normalization for GiGPO.
    Args:
        step_rewards: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.shape[-1]
    scores = step_rewards.clone()

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                print(f"id2score: {id2score}")
                print(f"len(id2score[idx]): {len(id2score[idx])}")
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if remove_std:
                scores[i] = scores[i] - id2mean[index[i]]
            else:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        step_advantages = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
    
    return step_advantages



def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def summarize_group_std_stats(id2std: Dict[Any, torch.Tensor],
                               id2score: Dict[Any, list],
                               idx2agent: Dict[Any, str] = None,
                               prefix: str = "adv_std",
                               thresholds: Sequence[float] = (1e-3, 1e-2, 1e-1)) -> Dict[str, float]:
    """
    Summarize the (pre-epsilon) group-level std statistics for wandb/console logging, so that we can
    empirically monitor whether per-agent (or per-group) std collapses to (near) zero during training
    -- i.e. the pathology Dr.GRPO originally identified with global std normalization, and that a
    per-agent std (Dr. MAS) could in principle reintroduce for small/near-homogeneous agent groups.

    Args:
        id2std: mapping from group key -> raw std (before the `+ epsilon` used in normalization)
        id2score: mapping from group key -> list of raw scores in that group (used for group size)
        idx2agent: optional mapping from group key -> agent_id string. If None, all groups are
            reported under a single "all" bucket (e.g. for vanilla GRPO with global grouping).
        prefix: metric name prefix
        thresholds: std thresholds used to report the fraction of (near-)degenerate groups

    Returns:
        A flat dict of scalar metrics, ready to be merged into the training `metrics` dict.
    """
    agent2stds = defaultdict(list)
    agent2sizes = defaultdict(list)
    for idx, std in id2std.items():
        agent = idx2agent[idx] if idx2agent is not None else "all"
        agent2stds[agent].append(float(std.item() if torch.is_tensor(std) else std))
        agent2sizes[agent].append(len(id2score[idx]))

    metrics = {}
    for agent, stds in agent2stds.items():
        stds_arr = np.array(stds, dtype=np.float64)
        sizes_arr = np.array(agent2sizes[agent], dtype=np.float64)
        metrics[f"{prefix}/{agent}/mean"] = float(np.mean(stds_arr))
        metrics[f"{prefix}/{agent}/min"] = float(np.min(stds_arr))
        metrics[f"{prefix}/{agent}/p5"] = float(np.percentile(stds_arr, 5))
        metrics[f"{prefix}/{agent}/median"] = float(np.median(stds_arr))
        metrics[f"{prefix}/{agent}/max"] = float(np.max(stds_arr))
        metrics[f"{prefix}/{agent}/num_groups"] = float(len(stds_arr))
        metrics[f"{prefix}/{agent}/group_size_mean"] = float(np.mean(sizes_arr))
        for thresh in thresholds:
            metrics[f"{prefix}/{agent}/frac_below_{thresh:g}"] = float(np.mean(stds_arr < thresh))
    return metrics


def _compute_global_uid_stats(raw_scores: torch.Tensor, uids: np.ndarray, traj_index: np.ndarray) -> tuple:
    """
    Mirrors vanilla GRPO's grouping exactly (see the `not group_by_agent_id` branch of
    `compute_grpo_outcome_advantage`): active steps within the same trajectory (possibly from
    different agents) are first averaged into a single scalar per trajectory, then mean/std are
    computed across trajectories sharing the same `uid`.

    This is used purely as a diagnostic "shadow" computation (see `summarize_group_diagnostics`)
    and never affects the actual advantage used for training.
    """
    bsz = raw_scores.shape[0]
    traj_acc = defaultdict(list)
    for i in range(bsz):
        traj_acc[(uids[i], traj_index[i])].append(raw_scores[i])
    id2score = defaultdict(list)
    for (u, _t), vals in traj_acc.items():
        id2score[u].append(torch.stack(vals).mean())
    id2mean, id2std = {}, {}
    for u, vals in id2score.items():
        if len(vals) == 1:
            id2mean[u], id2std[u] = torch.tensor(0.0), torch.tensor(1.0)
        else:
            t = torch.stack(vals)
            id2mean[u], id2std[u] = torch.mean(t), torch.std(t)
    return id2mean, id2std


def _compute_per_agent_stats(raw_scores: torch.Tensor, uids: np.ndarray, agent_ids: np.ndarray) -> tuple:
    """
    Mirrors Dr. MAS's grouping exactly (see the `group_by_agent_id` branch of
    `compute_grpo_outcome_advantage`): every active-step sample sharing the same (uid, agent_id) is
    pooled directly (no within-trajectory averaging), then mean/std are computed per (uid, agent_id).

    This is used purely as a diagnostic "shadow" computation (see `summarize_group_diagnostics`)
    and never affects the actual advantage used for training.
    """
    bsz = raw_scores.shape[0]
    id2score = defaultdict(list)
    idx2agent = {}
    for i in range(bsz):
        key = (uids[i], agent_ids[i])
        id2score[key].append(raw_scores[i])
        idx2agent[key] = str(agent_ids[i])
    id2mean, id2std = {}, {}
    for key, vals in id2score.items():
        if len(vals) == 1:
            id2mean[key], id2std[key] = torch.tensor(0.0), torch.tensor(1.0)
        else:
            t = torch.stack(vals)
            id2mean[key], id2std[key] = torch.mean(t), torch.std(t)
    return id2mean, id2std, idx2agent


def summarize_group_diagnostics(agent_mean: Dict[Any, torch.Tensor],
                                 agent_std: Dict[Any, torch.Tensor],
                                 idx2agent: Dict[Any, str],
                                 global_mean: Dict[Any, torch.Tensor],
                                 global_std: Dict[Any, torch.Tensor],
                                 prefix: str = "adv_diag") -> Dict[str, float]:
    """
    Connects Lemma 4.2's per-agent gradient-inflation term to directly loggable quantities, by
    comparing each agent's own (mu_k, sigma_k) (per (uid, agent_id) group) against the *global*
    (mu, sigma) that vanilla GRPO would have used for the very same uid (mu_k, sigma_k, mu, sigma
    are all computed independently of `group_by_agent_id`, so this can be logged whether the run is
    actually training with vanilla GRPO or with Dr. MAS).

    For every (uid, agent_id) group:
        mean_gap      = mu_k - mu
        var_ratio     = sigma_k^2 / sigma^2
        inflation     = (sigma_k^2 + mean_gap^2) / sigma^2      (the Lemma 4.2 multiplier)

    Returns a flat dict of per-agent aggregated statistics of the above three quantities.
    """
    agent2mean_gap = defaultdict(list)
    agent2var_ratio = defaultdict(list)
    agent2inflation = defaultdict(list)
    for key, mu_k_t in agent_mean.items():
        uid = key[0]
        if uid not in global_mean:
            continue
        mu_k = float(mu_k_t.item())
        sigma_k = float(agent_std[key].item())
        mu = float(global_mean[uid].item())
        sigma = float(global_std[uid].item())
        mean_gap = mu_k - mu
        var_ratio = (sigma_k**2) / (sigma**2 + 1e-12)
        inflation = (sigma_k**2 + mean_gap**2) / (sigma**2 + 1e-12)
        agent = idx2agent[key]
        agent2mean_gap[agent].append(mean_gap)
        agent2var_ratio[agent].append(var_ratio)
        agent2inflation[agent].append(inflation)

    metrics = {}
    for agent in agent2mean_gap:
        mg = np.array(agent2mean_gap[agent], dtype=np.float64)
        vr = np.array(agent2var_ratio[agent], dtype=np.float64)
        inf_ = np.array(agent2inflation[agent], dtype=np.float64)
        metrics[f"{prefix}/{agent}/mean_gap_mean"] = float(np.mean(mg))
        metrics[f"{prefix}/{agent}/mean_gap_abs_mean"] = float(np.mean(np.abs(mg)))
        metrics[f"{prefix}/{agent}/mean_gap_abs_max"] = float(np.max(np.abs(mg)))
        metrics[f"{prefix}/{agent}/var_ratio_mean"] = float(np.mean(vr))
        metrics[f"{prefix}/{agent}/var_ratio_min"] = float(np.min(vr))
        metrics[f"{prefix}/{agent}/var_ratio_max"] = float(np.max(vr))
        metrics[f"{prefix}/{agent}/inflation_factor_mean"] = float(np.mean(inf_))
        metrics[f"{prefix}/{agent}/inflation_factor_p95"] = float(np.percentile(inf_, 95))
        metrics[f"{prefix}/{agent}/inflation_factor_max"] = float(np.max(inf_))
    return metrics


def compute_loss_balance_weights(uids: np.ndarray, agent_ids: np.ndarray, device=None) -> torch.Tensor:
    """
    Per-sample loss weight that equalizes each (uid, agent_id) group's total contribution to the
    policy-gradient loss, regardless of how many active-step samples that agent happened to produce
    for that prompt (e.g. a verifier invoked 3x in a loop vs. 1x elsewhere). This isolates whether
    Dr. MAS's gain comes from *re-centering/re-scaling the advantage value* (mu_k, sigma_k) vs. from
    *implicitly re-weighting the gradient* towards agents/prompts with fewer active steps.

    weight_i = (1 / n_{(uid_i, agent_i)}) / mean(1 / n_{(uid_j, agent_j)} for all j)

    so that the batch-average weight is 1 (keeps the overall loss scale, and thus the effective
    learning rate, comparable to the unweighted baseline).

    Returns:
        loss_weights: `(torch.Tensor)`, shape (bs,)
    """
    bsz = len(uids)
    freq_key = list(zip(uids.tolist() if isinstance(uids, np.ndarray) else uids,
                         agent_ids.tolist() if isinstance(agent_ids, np.ndarray) else agent_ids))
    counts = Counter(freq_key)
    raw_weight = torch.tensor([1.0 / counts[k] for k in freq_key], dtype=torch.float32, device=device)
    return raw_weight / raw_weight.mean()


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
    agent_ids: np.ndarray = None,
    uids: np.ndarray = None,
    return_std_metrics: bool = False,
    balance_loss_by_agent_freq: bool = False,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.
        agent_ids: `(np.ndarray)`, optional
            shape is (bs,). Per-sample agent id, only used (if provided) to break the std-tracking
            metrics down by agent for logging purposes, and/or to compute the loss-balancing weights.
            Does not affect the advantage computation itself.
        uids: `(np.ndarray)`, optional
            shape is (bs,). Per-sample raw prompt-group uid (regardless of `group_by_agent_id`), only
            used (if provided, together with `agent_ids`) to compute the Lemma 4.2 diagnostics
            (see `summarize_group_diagnostics`) and/or the loss-balancing weights.
        return_std_metrics: bool
            If True, also return a dict of per-agent (raw, pre-epsilon) group-std statistics for
            monitoring whether std collapses to near zero during training (see `summarize_group_std_stats`),
            plus -- when both `uids` and `agent_ids` are provided -- the `adv_diag/*` mean-gap /
            var-ratio / inflation-factor diagnostics from `summarize_group_diagnostics`.
        balance_loss_by_agent_freq: bool
            If True, also return a per-sample `loss_weights` tensor (see `compute_loss_balance_weights`)
            that equalizes each (uid, agent_id) group's contribution to the policy loss. This is
            orthogonal to `group_by_agent_id`/`norm_adv_by_std_in_grpo` and can be combined with any
            of them, to isolate normalization effects from implicit invocation-frequency rebalancing.

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
        extra: `Dict[str, Any]`
            only returned when `return_std_metrics=True` or `balance_loss_by_agent_freq=True`.
            Contains `"std_metrics"` (Dict[str, float], possibly {}) and `"loss_weights"`
            (`torch.Tensor` of shape (bs,), or None).
    """
    print("group_by_agent_id: ", group_by_agent_id)
    scores = token_level_rewards.sum(dim=-1)
    want_extra = return_std_metrics or balance_loss_by_agent_freq

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    traj_accumulator = defaultdict(list)
    traj2avg = {}
    with torch.no_grad():
        bsz = scores.shape[0]
        # Snapshot of the raw per-sample rewards, taken before any in-place normalization below, so
        # that the diagnostics (which need the raw distribution) are unaffected by that mutation.
        raw_scores = scores.clone() if return_std_metrics else None

        for i in range(bsz):
            traj_accumulator[(index[i], traj_index[i])].append(scores[i])
        
        for (idx, t_idx), reward_list in traj_accumulator.items():
            if group_by_agent_id:
                id2score[idx].extend(reward_list)
            else:
                avg_score = torch.stack(reward_list).mean()
                traj2avg[(idx, t_idx)] = avg_score
                id2score[idx].append(avg_score)
        if not group_by_agent_id:
            for i in range(bsz):
                scores[i] = traj2avg[(index[i], traj_index[i])]

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        std_metrics = {}
        if return_std_metrics:
            # Only break the adv_std/* bucketing down by agent when grouping is itself agent-aware
            # (group_by_agent_id=True); otherwise `index` mixes multiple agents per group and
            # labeling it by whichever agent happens to appear first would be misleading.
            idx2agent = None
            if agent_ids is not None and group_by_agent_id:
                idx2agent = {}
                for i in range(bsz):
                    idx2agent.setdefault(index[i], str(agent_ids[i]))
            std_metrics = summarize_group_std_stats(id2std, id2score, idx2agent=idx2agent)

            if uids is not None and agent_ids is not None:
                # Lemma 4.2 diagnostics: connect the theoretical inflation term to a loggable metric,
                # computed independently of `group_by_agent_id` so it can be reported for either a
                # vanilla-GRPO or a Dr. MAS training run.
                global_mean, global_std = _compute_global_uid_stats(raw_scores, uids, traj_index)
                agent_mean, agent_std, diag_idx2agent = _compute_per_agent_stats(raw_scores, uids, agent_ids)
                diag_metrics = summarize_group_diagnostics(agent_mean, agent_std, diag_idx2agent, global_mean, global_std)
                std_metrics.update(diag_metrics)

        loss_weights = None
        if balance_loss_by_agent_freq:
            if uids is None or agent_ids is None:
                raise ValueError("balance_loss_by_agent_freq=True requires both `uids` and `agent_ids` to be provided.")
            loss_weights = compute_loss_balance_weights(uids, agent_ids, device=scores.device)

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

        if return_std_metrics:
            # Tail behavior of the *normalized* per-sample advantage (post +epsilon), broken down by
            # agent. This directly shows whether small std groups translate into blown-up advantages.
            agent2abs_adv = defaultdict(list)
            for i in range(bsz):
                agent = str(agent_ids[i]) if agent_ids is not None else "all"
                agent2abs_adv[agent].append(abs(float(scores[i].item())))
            for agent, vals in agent2abs_adv.items():
                vals_arr = np.array(vals, dtype=np.float64)
                std_metrics[f"adv_norm/{agent}/abs_max"] = float(np.max(vals_arr))
                std_metrics[f"adv_norm/{agent}/abs_p99"] = float(np.percentile(vals_arr, 99))
                std_metrics[f"adv_norm/{agent}/abs_mean"] = float(np.mean(vals_arr))

        scores = scores.unsqueeze(-1) * response_mask

    if want_extra:
        return scores, scores, {"std_metrics": std_metrics, "loss_weights": loss_weights}
    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    group_by_agent_id: bool = False,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group
        group_by_agent_id: bool
            If True, the mean and std are computed across agent group.
            If False (i.e., standard episode-level adv), the mean and std are computed across trajectories within one group.

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, traj_index: np.ndarray, epsilon: float = 1e-6, group_by_agent_id: bool = False):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    seen_pairs = set()
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not group_by_agent_id:
                seen_pairs.add((index[i], traj_index[i]))
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    loss_weights=None,
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        loss_weights (torch.Tensor, optional):
            Per-sample weight, shape (batch_size,) or (batch_size, response_length). If provided, it
            is multiplied into the (unclipped/clipped) per-token pg loss *before* aggregation, e.g. to
            equalize each agent/prompt's contribution to the loss regardless of its invocation
            frequency (see `compute_loss_balance_weights`). It does NOT affect `pg_clipfrac`/`ppo_kl`,
            which remain raw/unweighted diagnostics.
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    if loss_weights is not None:
        if loss_weights.dim() == 1:
            loss_weights = loss_weights.unsqueeze(-1)
        pg_losses = pg_losses * loss_weights
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
