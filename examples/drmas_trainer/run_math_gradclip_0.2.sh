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
# Stabilization-alternative baseline: VANILLA GRPO + stronger gradient clipping (grad_clip=0.2),
# one of three grad_clip values swept as an alternative to Dr. MAS
# (see run_math_gradclip_0.5.sh, run_math_gradclip_0.1.sh, and run_math_std_tracking.sh for Dr. MAS).
# grad_clip is overridden identically for every agent here via agent_specific_parameters (which also
# supports giving each agent a *different* grad_clip if desired).

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
group_by_agent_id=False  # vanilla GRPO -- an alternative to Dr. MAS, not a combination with it
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
actor_grad_clip='[0.2,0.2]'

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)

experiment_name="drmas_gradclip0.2_${model_name_tag}"

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
    actor_rollout_ref.actor.grad_clip=null \
    +agent.agent_specific_parameters.actor.grad_clip=$actor_grad_clip \
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
