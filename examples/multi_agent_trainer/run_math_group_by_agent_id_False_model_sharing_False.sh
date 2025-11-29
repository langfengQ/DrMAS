set -x

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test.parquet" # Full test dataset
    train_data_size=32
    val_data_size=512
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_math/train.parquet"
    VAL_DATA="$HOME/data/drmas_math/test_sampled.parquet" # For fast validation during training (test_sampled.parquet contains 50 examples from MATH500, 30 examples from AIME2024, and 30 examples from AIME2025)
    train_data_size=32
    val_data_size=110
fi

###################### Algorithm Configurations #################
algorithm=grpo
group_by_agent_id=False

##################### Agent Configurations #####################
agent_ids='["Solver Agent","Verifier Agent"]'
model_ids='["Qwen/Qwen3-8B","Qwen/Qwen3-8B"]'
model_sharing=False

orchestra_type=math
max_loop_num=2

# Agent-specific parameter override (only support actor_rollout_ref)
actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[2,2]'

##################### Training Configurations #################
group_size=8
ppo_mini_update_num=1

max_prompt_length=8192
max_response_length=4096

####################### Other Configurations #####################

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)

experiment_name="drmas${group_by_agent_id}_share${model_sharing}_${model_name_tag}"
default_local_dir="/mnt/hdfs/tiktok_aiic/user/longtao.zheng/multiagent_checkpoints/${experiment_name}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    data.truncation='middle' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=$ppo_mini_update_num \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    agent.orchestra.math.max_loop_num=$max_loop_num \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DrMAS_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.default_local_dir="$default_local_dir" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True
