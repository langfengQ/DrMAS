set -x
ENGINE=${1:-vllm}

multi_agent=True
agent_list='["Action Agent","Memory Agent"]'

train_data_size=256
val_data_size=512
group_size=8

algorithm=grpo
gigpo_mode="mean_std_norm" # "mean_norm" or "mean_std_norm"
model=Qwen/Qwen2.5-1.5B-Instruct

if [ "$multi_agent" = "True" ]; then
    num_agents=$(echo "$agent_list" | jq length)
    agent_name_tag=$(echo "$agent_list" | jq -r '.[]' | sed 's/ Agent//g' | paste -sd+ -)
else
    num_agents=1
    agent_name_tag="Single"
fi

ppo_mini_batch_size=$((num_agents * 128)) # dynamically compute batch size to ensure the stable update times per interation
experiment_name="${algorithm}_$(basename $model)_${group_size}group_${agent_name_tag}"

echo "Agent List: $agent_list"
echo "Number of Agents: $num_agents"
echo "ppo_mini_batch_size: $ppo_mini_batch_size"
echo "agent_name_tag: $agent_name_tag"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$HOME/data/searchR1_processed_direct/train.parquet \
    data.val_files=$HOME/data/searchR1_processed_direct/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=10240 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=10240 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.sim_thresh=0.5 \
    algorithm.gigpo.mode=$gigpo_mode \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=4 \
    env.rollout.n=$group_size \
    env.history_length=4 \
    agent.agent_list="$agent_list" \
    agent.multi_agent=$multi_agent \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@