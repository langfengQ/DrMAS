```bash
# bash setup.sh

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/tiger/test/miniconda
unset PYTHONPATH
source /opt/tiger/test/miniconda/bin/activate
rm Miniconda3-latest-Linux-x86_64.sh

conda create -n multiagent python==3.12
conda activate multiagent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .

pip install -r requirements_sglang.txt
pip install gym==0.26.2 --user

# pip install flash-attn==2.7.4.post1 --no-build-isolation --user
# pip install -e . --user
# pip install -r requirements_sglang.txt --user
# pip install gym==0.26.2 --user
# pip install protobuf==3.20.1 --user

# cd ./agent_system/environments/env_package/search/third_party
# pip install -e . --user

mkdir ~/data
hdfs dfs -get hdfs://harunava/home/byte_malia_gcp_aiic/user/zhenghai.xue/search_r1 ~/data
# cd ~/data
# cp -r search_r1/searchR1_processed_direct .

cd ~/data/search_r1
unzip retriever.zip

mv ~/data/search_r1 ~/data/searchR1
```

```bash
cd /mnt/bn/tiktok-mm-5/aiic/users/longtao.zheng/code/multiagent
python examples/data_preprocess/preprocess_search_r1_dataset.py

source ~/data/searchR1/bin/activate
cd /mnt/bn/tiktok-mm-5/aiic/users/longtao.zheng/code/multiagent
# pip install -U cffi soundfile --user
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log &

source ~/data/searchR1/bin/deactivate
# bash examples/multi_agent_trainer/run_search.sh

source /opt/tiger/test/miniconda/bin/activate
conda activate multiagent
cd /mnt/bn/tiktok-mm-5/aiic/users/longtao.zheng/code/multiagent

export WANDB_API_KEY=d65108a54043949f2e975c92c7d2089c414439bf
VERL_MASTER_PORT=32800 bash examples/multi_agent_trainer/run_search_0_50.sh
```
