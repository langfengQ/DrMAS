set -x
ENGINE=${1:-vllm}
export CUDA_VISIBLE_DEVICES=4,5

multi_agent=True
# agent_list='["Search Agent"]'
agent_list='["Search Agent", "Verify Agent"]'
executor_type=search
use_agent_memory=False

train_data_size=256
val_data_size=512
group_size=8

algorithm=grpo
gigpo_mode=mean_std_norm # "mean_norm" or "mean_std_norm"
model=Qwen/Qwen2.5-3B-Instruct

if [ "$multi_agent" = "True" ]; then
    agent_name_tag=$(echo "$agent_list" | jq -r '.[]' | sed 's/ Agent//g' | paste -sd+ -)
else
    agent_name_tag="Single"
fi

experiment_name="${algorithm}_$(basename $model)_${group_size}group_${agent_name_tag}"


TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=5120 \
    data.max_response_length=500 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=10 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.985 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.sim_thresh=0.8 \
    algorithm.gigpo.mode=$gigpo_mode \
    env.env_name=search \
    env.seed=0 \
    env.max_steps=3 \
    env.rollout.n=$group_size \
    env.history_length=3 \
    agent.multi_agent=$multi_agent \
    agent.agent_list="$agent_list" \
    agent.executor_type=$executor_type \
    agent.use_agent_memory=$use_agent_memory \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='multiagent_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=150 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@