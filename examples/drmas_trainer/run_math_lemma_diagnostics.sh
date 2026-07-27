# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Nanyang Technological University (NTU), Singapore
# Copyright 2025 verl-agent (GiGPO) Team
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
#
# Lemma 4.2 diagnostics run: connects the theoretical per-agent gradient-inflation term directly to
# loggable training-time quantities, so we can check whether inflation spikes line up with observed
# clip-ratio / KL / grad-norm spikes (reviewer request: "report empirical measurements of the
# proposed instability factors... to connect the theoretical analysis more directly to the observed
# gradient spikes").
#
# This defaults to VANILLA GRPO (group_by_agent_id=False), i.e. the setting where the instability
# is claimed to occur, so we can see whether the diagnostics actually predict it. The diagnostics
# themselves (see verl/trainer/ppo/core_algos.py::summarize_group_diagnostics) are computed from a
# self-contained "shadow" (mu, sigma) / (mu_k, sigma_k) calculation that is independent of
# group_by_agent_id, so simply flipping that flag to True (i.e. re-using run_math_std_tracking.sh)
# lets you compare the same diagnostics under Dr. MAS training instead.
#
# New metrics logged every training step, on top of the existing per-agent
# actor/<wg_id>/{pg_clipfrac,ppo_kl,grad_norm} and adv_std/<agent>/* (from run_math_std_tracking.sh):
#   adv_diag/<agent>/mean_gap_mean, mean_gap_abs_mean, mean_gap_abs_max
#       mu_k - mu, i.e. how far this agent's own active-step mean reward is from the global
#       (uid-only) mean that vanilla GRPO would have centered on.
#   adv_diag/<agent>/var_ratio_mean, var_ratio_min, var_ratio_max
#       sigma_k^2 / sigma^2
#   adv_diag/<agent>/inflation_factor_mean, inflation_factor_p95, inflation_factor_max
#       (sigma_k^2 + mean_gap^2) / sigma^2 -- the exact Lemma 4.2 gradient-norm multiplier.
#
# Suggested analysis: export adv_diag/<agent>/inflation_factor_* together with
# actor/<agent_wg_id>/{pg_clipfrac,ppo_kl,grad_norm} from wandb and check whether inflation spikes
# precede/coincide with clip-ratio, KL, or grad-norm spikes for the same agent.

set -x

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test.parquet" # Full test dataset
    train_data_size=32
    val_data_size=64
    val_group_size=16  # For pass@16 and avg@16 computation during evaluation
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test_sampled.parquet" # For fast validation during training (test_sampled.parquet contains 50 examples from MATH500, 30 examples from AIME2024, and 30 examples from AIME2025)
    train_data_size=32
    val_data_size=110
    val_group_size=1
fi

###################### Algorithm Configurations #################
algorithm=grpo
group_size=8
group_by_agent_id=False       # vanilla GRPO -- the setting whose instability we want to diagnose
norm_adv_by_std_in_grpo=True

##################### Agent Configurations #####################
agent_ids='["Solver Agent","Verifier Agent"]'
model_ids='["Qwen/Qwen3-4B","Qwen/Qwen3-4B"]'
model_sharing=False

orchestra_type=math
max_loop_num=2

# Agent-specific parameter override (only support actor_rollout_ref)
actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[4,4]'

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)

experiment_name="drmas_lemmadiag_share${model_sharing}_${model_name_tag}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=8192 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    algorithm.norm_adv_by_std_in_grpo=$norm_adv_by_std_in_grpo \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.math.max_loop_num=$max_loop_num \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DrMAS_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True
