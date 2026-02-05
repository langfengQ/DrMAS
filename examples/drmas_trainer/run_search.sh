set -x

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
    TRAIN_DATA="$HOME/data/drmas_search/train.parquet"
    VAL_DATA="$HOME/data/drmas_search/test.parquet" # Full test dataset
    train_data_size=128
    val_data_size=64
    val_group_size=16  # For pass@16 and avg@16 computation during evaluation
else
    echo "Running in training mode"
    VAL_ONLY=False
    TRAIN_DATA="$HOME/data/drmas_search/train.parquet"
    VAL_DATA="$HOME/data/drmas_search/test_sampled.parquet" # For fast validation during training (sample 30 entries per data source, total 210 samples)
    train_data_size=128
    val_data_size=256
    val_group_size=1
fi

###################### Algorithm Configurations #################
algorithm=grpo
group_size=5
group_by_agent_id=True # enable Dr. MAS

##################### Agent Configurations #####################
agent_ids='["Verifier Agent","Search Agent","Answer Agent"]'
model_ids='["Qwen/Qwen2.5-7B","Qwen/Qwen2.5-7B","Qwen/Qwen2.5-7B"]'
model_sharing=False

orchestra_type=search
max_turn=4

# Agent-specific parameter override (only support actor_rollout_ref)
actor_optim_lr='[1e-6,1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[4,4,4]'

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_' | paste -sd_)

experiment_name="drmas_share${model_sharing}_${model_name_tag}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=800 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=5 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.group_by_agent_id=$group_by_agent_id \
    env.env_name=search \
    env.seed=0 \
    env.search.search_url='http://127.0.0.1:8000/retrieve' \
    env.max_steps=$max_turn \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    env.history_length=$max_turn \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='DrMAS_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True
