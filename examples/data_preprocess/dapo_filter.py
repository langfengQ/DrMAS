
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
    parser.add_argument("--local_dir", default="~/data/dapo_filter")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "dapo_filter"

    train_dataset = datasets.load_dataset("aaabiao/dapo_filter", split="train")
    test_dataset = datasets.load_dataset("aaabiao/dapo_filter", split="train")

    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    #     r"The final answer MUST BE put in \boxed{}."
    # )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        _rm_suffix = re.compile(
            r"(\r?\n|\\n)*\s*Let'?s\s+think\s+step\s+by\s+step\s+and\s+output\s+the\s+final\s+answer\s+within\s*\\boxed\{\}\.\s*$",
            re.IGNORECASE
        )
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            reward_model = example.pop("reward_model")
            ability = example.pop("ability")
            assert len(prompt) == 1
            prompt = prompt[0]["content"]
            question = _rm_suffix.sub("", prompt)
            assert "Let's think step by step and output" not in question
            ground_truth = reward_model["ground_truth"]
            print(f"ground_truth: {ground_truth}")
            data = {
                "data_source": data_source,
                "ability": ability,
                "reward_model": reward_model,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": ground_truth,
                    "question": question,
                },
                "env_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source}

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
