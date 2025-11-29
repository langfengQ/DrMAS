```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install -e .
pip install -r requirements_sglang.txt --user
pip install gym==0.26.2 --user
pip install peft==0.17.1 --user

export WANDB_API_KEY=ed5069227da5d2bfc22ddd654a7f3a2b87475c1f

python examples/data_preprocess/drmas_math.py

bash examples/multi_agent_trainer/run_math_group_by_agent_id_True_model_sharing_True.sh
```
