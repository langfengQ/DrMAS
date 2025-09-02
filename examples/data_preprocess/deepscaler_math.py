
"""
Preprocess the DeepScaler dataset to parquet format
"""

import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/data/langfeng/data/deepscaler_math")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "deepscaler_math"

    train_dataset = datasets.load_dataset("agentica-org/DeepScaleR-Preview-Dataset", split="train")
    test_dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")

    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    #     r"The final answer MUST BE put in \boxed{}."
    # )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("problem")
            solution = example.pop("answer")
            prompt = question

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "ability": "deepscaler_math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                },
                "env_kwargs": {"ground_truth": solution, "question": question, "data_source": data_source}

            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
