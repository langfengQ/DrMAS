set -x
export CUDA_VISIBLE_DEVICES=2,3,4,5

agent_ids='["Math Agent"]'
# "Reflexion Agent"
# "Search Agent"
# "Critic Agent"

model_ids='["Qwen/Qwen2.5-7B-Instruct"]'
model_sharing=False
random_dropout=False
random_dropout_ratio=0.5

# "meta-llama/Llama-3.2-3B-Instruct"
# "Qwen/Qwen3-4B-Instruct-2507"
# "Qwen/Qwen2.5-1.5B-Instruct"

orchestra_type=math
use_agent_memory=False

train_data_size=256
val_data_size=512
group_size=16
ppo_mini_update_num=16

algorithm=grpo

max_prompt_length=2000
max_response_length=4000

model_name_tag=$(jq -r '.[]' <<< "$model_ids"  | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')
agent_name_tag=$(jq -r '.[]' <<< "$agent_ids" | sed 's/ Agent//g' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

combined_tag=$(paste -d_ <(echo "$agent_name_tag") <(echo "$model_name_tag") | paste -sd_ -)

dropout_tag=$([[ "${random_dropout,,}" == "true" ]] && printf 'dropout%s' "${random_dropout_ratio}" || printf '%s' 'nodropout')

experiment_name="${algorithm}_bs${train_data_size}_${group_size}group_${ppo_mini_update_num}update_${max_prompt_length}prompt_${max_response_length}res_${dropout_tag}_share${model_sharing}_${combined_tag}"
# experiment_name="${algorithm}_bs${train_data_size}_${group_size}group_${ppo_mini_update_num}update_${max_prompt_length}prompt_${max_response_length}res_${dropout_tag}_share${model_sharing}_qwen2.5_3b_it"

TRAIN_DATA="/mnt/raid/data/langf/data/deepscaler_math/train.parquet"
VAL_DATA="/mnt/raid/data/langf/data/deepscaler_math/test.parquet"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=None \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=$ppo_mini_update_num \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.01 \
    algorithm.use_kl_in_reward=False \
    env.env_name=math \
    env.seed=0 \
    env.rollout.n=$group_size \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.orchestra_type=$orchestra_type \
    agent.use_agent_memory=$use_agent_memory \
    agent.random_dropout=$random_dropout \
    agent.random_dropout_ratio=$random_dropout_ratio \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='multiagent_math' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=20 \
    trainer.total_epochs=10 \
    trainer.val_before_train=False $@