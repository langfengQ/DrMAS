# Installation
## Install veRL
```bash
conda create -n multiagent python==3.12 -y
conda activate multiagent

pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation

pip3 install -e .

pip3 install -r requirements_sglang.txt
```

## Install Supported Environments
### 1. Search
```bash
conda activate multiagent
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
For fast validation during the training, sample 30 entries from each data source (total 210 samples):
```bash
cd repo_root/

# For fast validation during training (sample 30 entries per data source, total 210 samples):
python examples/data_preprocess/preprocess_search_r1_dataset.py --samples_per_source 30

# Or, to process the full test dataset:
# python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

### 2. Math
Prepare the dataset (test data contains 50 examples from MATH500 and 30 examples from AIME2024):
```bash
cd repo_root/
python examples/data_preprocess/dapo_filter.py
```

## Training Scrpts

### 1. Search
```bash
bash examples/multi_agent_trainer/run_search.sh
```

### 2. Math
```bash
bash examples/multi_agent_trainer/run_math.sh
```
