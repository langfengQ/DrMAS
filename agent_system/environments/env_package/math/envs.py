import asyncio
import concurrent.futures
from typing import Any, Dict, List
from agent_system.environments.env_package.math.math_reward import RewardConfig, RewardMathFn

class MathEnv:
    """
    Environment for Math execution tasks.
    """

    def __init__(self,):
        self.ground_truth = None
        self.data_source = "unknown"
        reward_config = RewardConfig()
        self.reward_fn = RewardMathFn(reward_config)


    def reset(self, extras: Dict[str, Any]) -> str:
        self.ground_truth = extras["ground_truth"]
        self.data_source = extras.get("data_source", "unknown")

    def step(self, action: str):
        done = True  # always done after one step
        task_info = {
            "ground_truth": self.ground_truth,
            "has_toolcall": False,
        }
        reward, is_correct = self.reward_fn(task_info, action)
        obs = None
        info = {"data_source": self.data_source, "won": is_correct}
        return obs, reward, done, info

    def close(self) -> None:
        pass

class MathMultiProcessEnv:
    """
    A simple multi-environment wrapper running MathEnv in parallel threads.

    - env_num: logical number of groups (kept for external compatibility)
    - group_n: envs per group
    - total_envs = env_num * group_n
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.env_num   = env_num
        self.group_n   = group_n
        self.batch_size = env_num * group_n
        self.is_train  = is_train

        self.envs = [MathEnv() for _ in range(self.batch_size)]

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _sync_reset(self, env, kwargs):
        extras = {
            "ground_truth": kwargs["ground_truth"],
            "data_source": kwargs.get("data_source", "unknown")
        }
        env.reset(extras)
        obs = kwargs["question"]
        info = {'data_source': kwargs.get("data_source", "unknown")}
        return obs, info
    
    def _sync_step(self, env, action: str):
        obs, reward, done, info = env.step(action)
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
                    "ground_truth": "",
                    "question": "",
                    "data_source": "unkown",
                }


        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, info_list

    def step(self, actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act)
            for env, act in zip(self.envs, padded_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_math_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
):
    return MathMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )