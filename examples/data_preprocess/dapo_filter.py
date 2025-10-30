
"""
Preprocess the DeepScaler dataset to parquet format
"""

import argparse
import os
import re

import datasets
from datasets import concatenate_datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/dapo_filter")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    train_dataset = datasets.load_dataset("aaabiao/dapo_filter", split="train")
    test_dataset_aime24 = datasets.load_dataset("HuggingFaceH4/aime_2024", split="train")
    test_dataset_math500 = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    test_dataset_math500 = test_dataset_math500.select(range(50))
    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. "
    #     r"The final answer MUST BE put in \boxed{}."
    # )

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
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

    def make_map_fn_test(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem")
            ability = "math"
            ground_truth = example.pop("answer")
            data = {
                "data_source": data_source,
                "ability": ability,
                "reward_model": {"style": "rule", "ground_truth": ground_truth},
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

    train_dataset = train_dataset.map(function=make_map_fn("train", "dapo_filter"), with_indices=True)
    test_dataset_aime24 = test_dataset_aime24.map(function=make_map_fn_test("test", "aime24"), with_indices=True)
    test_dataset_math500 = test_dataset_math500.map(function=make_map_fn_test("test", "math500"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    test_dataset = concatenate_datasets([test_dataset_aime24, test_dataset_math500])
    print(f"Combined test_dataset length: {len(test_dataset)}")

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
