set -x

export CUDA_VISIBLE_DEVICES=2,3,4,5

agent_ids='["Reflexion Agent", "Search Agent"]'
# agent_ids='["Reflexion Agent", "Search Agent", "Verify Agent"]'

agent_models='["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]'

# "meta-llama/Llama-3.2-3B-Instruct"
# "Qwen/Qwen3-4B-Instruct-2507"
# "Qwen/Qwen2.5-1.5B-Instruct"

executor_type=search
use_agent_memory=False

train_data_size=256
val_data_size=512
group_size=5
max_turn=3
ppo_mini_update_num=10

algorithm=grpo
gigpo_mode=mean_std_norm # "mean_norm" or "mean_std_norm"

max_prompt_length=4096
max_response_length=1024



model_name_tag=$(jq -r '.[]' <<< "$agent_models"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
agent_name_tag=$(jq -r '.[]' <<< "$agent_ids" | sed 's/ Agent//g' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

combined_tag=$(paste -d_ <(echo "$agent_name_tag") <(echo "$model_name_tag") | paste -sd_ -)

experiment_name="${algorithm}_bs${train_data_size}_${group_size}group_${max_turn}turn_${ppo_mini_update_num}update_${max_prompt_length}prompt_${max_response_length}res_${combined_tag}"


TRAIN_DATA="$HOME/data/searchR1_processed_direct/train.parquet"
VAL_DATA="$HOME/data/searchR1_processed_direct/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=None \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=$ppo_mini_update_num \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
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
    env.search.search_url='http://172.21.25.61:7856/retrieve' \
    env.max_steps=$max_turn \
    env.rollout.n=$group_size \
    env.history_length=$max_turn \
    agent.agent_ids="$agent_ids" \
    agent.agent_models="$agent_models" \
    agent.executor_type=$executor_type \
    agent.use_agent_memory=$use_agent_memory \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='multiagent_search' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=150 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@