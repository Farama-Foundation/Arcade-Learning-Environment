"""Wrappers for AtariVectorEnv: episode statistics, reward transform, and torch ops."""

from __future__ import annotations

import collections
import time
from typing import Union

import numpy as np
from gymnasium.logger import warn
from gymnasium.vector.vector_env import AutoresetMode

from ale_py.vector_env import AtariVectorEnv


class AtariVectorEnvWrapper:
    """Base wrapper for AtariVectorEnv that preserves send/recv API."""

    def __init__(self, env: Union[AtariVectorEnv, "AtariVectorEnvWrapper"]):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, actions, gamma=1.0, paddle_strength=1.0):
        return self.env.step(actions, gamma=gamma, paddle_strength=paddle_strength)

    def send(self, actions, gamma=1.0, paddle_strength=1.0):
        self.env.send(actions, gamma=gamma, paddle_strength=paddle_strength)

    def recv(self):
        return self.env.recv()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def test_atari_vector_env_wrapper():
    env = AtariVectorEnv(game="pong", num_envs=1)
    wrapper = AtariVectorEnvWrapper(env)

    obs, info = wrapper.reset()
    assert obs.shape == (1, 4, 84, 84)

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    assert obs.shape == (1, 4, 84, 84)

    wrapper.send(action)
    obs, reward, terminated, truncated, info = wrapper.recv()
    assert obs.shape == (1, 4, 84, 84)

    wrapper.close()
    assert env.closed


class _PerEnvMeanBuffer:
    """Per-environment rolling mean buffer."""

    def __init__(self, num_envs: int, capacity: int):
        self._capacity = capacity
        self._queues = [collections.deque(maxlen=capacity) for _ in range(num_envs)]

    def add(self, env_idx: int, val) -> None:
        self._queues[env_idx].append(val)

    def mean_all(self) -> np.ndarray:
        return np.array([
            float(np.mean(q)) if q else np.nan
            for q in self._queues
        ])


class RecordEpisodeStatistics(AtariVectorEnvWrapper):
    """Records cumulative rewards and episode lengths; adds them to info under 'episode'."""

    def __init__(self, env, buffer_length: int = 100, stats_key: str = "episode"):
        super().__init__(env)
        self._stats_key = stats_key
        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` in metadata; "
                "assuming AutoresetMode.NEXT_STEP."
            )
            self._autoreset_mode = AutoresetMode.NEXT_STEP
        else:
            assert isinstance(self.env.metadata["autoreset_mode"], AutoresetMode)
            self._autoreset_mode = self.env.metadata["autoreset_mode"]

        self.episode_count = 0
        self.episode_start_times = np.zeros((self.num_envs,))
        self.episode_returns = np.zeros((self.num_envs,))
        self.episode_lengths = np.zeros((self.num_envs,), dtype=int)
        self.prev_dones = np.zeros((self.num_envs,), dtype=bool)

        self.time_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)
        self.return_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)
        self.length_queue = _PerEnvMeanBuffer(self.num_envs, buffer_length)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        if options is not None and "reset_mask" in options:
            reset_mask = options.pop("reset_mask")
            assert isinstance(reset_mask, np.ndarray)
            assert reset_mask.shape == (self.num_envs,)
            assert reset_mask.dtype == np.bool_
            assert np.any(reset_mask)
            self.episode_start_times[reset_mask] = time.perf_counter()
            self.episode_returns[reset_mask] = 0
            self.episode_lengths[reset_mask] = 0
            self.prev_dones[reset_mask] = False
        else:
            self.episode_start_times = np.full(self.num_envs, time.perf_counter())
            self.episode_returns = np.zeros(self.num_envs)
            self.episode_lengths = np.zeros(self.num_envs, dtype=int)
            self.prev_dones = np.zeros(self.num_envs, dtype=bool)

        return obs, info

    def _update_episode_statistics(self, rewards, terminations, truncations, infos):
        assert isinstance(infos, dict), (
            f"`RecordEpisodeStatistics` requires info type to be dict, got {type(infos)}."
        )

        self.episode_returns[self.prev_dones] = 0
        self.episode_returns[~self.prev_dones] += rewards[~self.prev_dones]

        self.episode_lengths[self.prev_dones] = 0
        steps = infos.get("steps_taken", np.ones(self.num_envs, dtype=int))
        self.episode_lengths[~self.prev_dones] += steps[~self.prev_dones]

        self.episode_start_times[self.prev_dones] = time.perf_counter()

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = int(np.sum(dones))

        if not num_dones:
            return

        if self._stats_key in infos or f"_{self._stats_key}" in infos:
            raise ValueError(
                f"Key '{self._stats_key}' already exists in info: {list(infos.keys())}"
            )

        episode_time_length = np.round(time.perf_counter() - self.episode_start_times, 6)

        for i in np.where(dones)[0]:
            self.time_queue.add(i, float(episode_time_length[i]))
            self.return_queue.add(i, float(self.episode_returns[i]))
            self.length_queue.add(i, int(self.episode_lengths[i]))

        infos[self._stats_key] = {
            "r": np.where(dones, self.episode_returns, 0.0),
            "l": np.where(dones, self.episode_lengths, 0),
            "t": np.where(dones, episode_time_length, 0.0),
            "avg_r": self.return_queue.mean_all(),
            "avg_l": self.length_queue.mean_all(),
            "avg_t": self.time_queue.mean_all(),
        }
        infos[f"_{self._stats_key}"] = dones
        self.episode_count += num_dones

    def step(self, actions, gamma=1.0, paddle_strength=1.0):
        obs, rewards, terminations, truncations, infos = self.env.step(
            actions, gamma=gamma, paddle_strength=paddle_strength
        )
        self._update_episode_statistics(rewards, terminations, truncations, infos)
        return obs, rewards, terminations, truncations, infos

    def recv(self):
        obs, rewards, terminations, truncations, infos = super().recv()
        self._update_episode_statistics(rewards, terminations, truncations, infos)
        return obs, rewards, terminations, truncations, infos


class TransformReward(AtariVectorEnvWrapper):
    """Transform rewards using a given function (e.g. np.sign for reward clipping)."""

    def __init__(self, env, transform_fn=np.sign):
        super().__init__(env)
        self.transform_fn = transform_fn

    def recv(self):
        obs, rewards, terminations, truncations, infos = super().recv()
        rewards = self.transform_fn(rewards)
        return obs, rewards, terminations, truncations, infos


class TorchOpsWrapper(AtariVectorEnvWrapper):
    """Registers ALE torch custom ops; ale_recv routes through the Python wrapper chain.

    Wrappers that override step() but not recv() will be skipped during ale_recv
    and trigger a warning at construction time.
    """

    def __init__(self, env):
        super().__init__(env)
        (
            self.handle_id,
            self.ale_send,
            self.ale_step,
            _,
            self._unregister,
        ) = env.torch()

        e = env
        while hasattr(e, "env"):
            cls = type(e)
            if "step" in cls.__dict__ and "recv" not in cls.__dict__:
                warn(
                    f"{cls.__name__} overrides step() but not recv(). "
                    "Its processing will be skipped during ale_recv; "
                    "add a recv() override to support the async send/recv API."
                )
            e = e.env

    def ale_recv(self, handle_id: int):
        import torch

        obs, reward, term, trunc, infos = self.env.recv()
        return (
            torch.as_tensor(obs),
            torch.as_tensor(reward),
            torch.as_tensor(term),
            torch.as_tensor(trunc),
            infos,
        )

    def close(self):
        self._unregister()
        super().close()


def test_torch_ops_wrapper():
    import torch

    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep")
    wrapped = TorchOpsWrapper(env)
    wrapped.reset()

    actions = torch.zeros(1, dtype=torch.int64)
    wrapped.ale_send(wrapped.handle_id, actions)
    obs, rewards, _terms, _truncs, _infos = wrapped.ale_recv(wrapped.handle_id)
    assert obs.shape == (1, 4, 84, 84)
    assert rewards.shape == (1,)

    wrapped.close()

    from ale_py._torch_ops import _torch_envs
    assert wrapped.handle_id not in _torch_envs


def test_torch_ops_wrapper_episode_stats():
    """ale_recv must update RecordEpisodeStatistics and apply TransformReward."""
    import torch

    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep", reward_clipping=False)
    env = RecordEpisodeStatistics(env)
    env = TransformReward(env, np.sign)
    wrapped = TorchOpsWrapper(env)

    wrapped.reset()
    actions = torch.zeros(1, dtype=torch.int64)
    episode_completed = False
    for _ in range(10000):
        wrapped.ale_send(wrapped.handle_id, actions)
        obs, reward, term, trunc, infos = wrapped.ale_recv(wrapped.handle_id)
        assert float(reward.abs().max()) <= 1.0, f"Reward not clipped: {reward}"
        if "episode" in infos:
            episode_completed = True
            assert "r" in infos["episode"]
            assert "l" in infos["episode"]
            break
    assert episode_completed, "No episode completed in 10000 steps"
    wrapped.close()


# --- dummy envs for unit tests ---

class _DummyVectorEnv:
    def __init__(self, rewards: np.ndarray):
        assert rewards.ndim == 2
        self._rewards = rewards
        self.num_envs = rewards.shape[1]
        self._t = 0
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8), {}

    def step(self, actions, gamma=1.0, paddle_strength=1.0):
        obs = np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)
        reward = self._rewards[self._t]
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        if self._t == self._rewards.shape[0] - 1:
            terminated[:] = True
        self._t += 1
        return obs, reward, terminated, truncated, {}


class _DummyRecvEnv:
    def __init__(self, rewards: np.ndarray):
        assert rewards.ndim == 1
        self._rewards = rewards
        self.num_envs = rewards.shape[0]

    def recv(self):
        obs = np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)
        return obs, self._rewards.copy(), np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), {}


def test_record_episode_statistics_with_clipped_vs_raw_rewards():
    rewards = np.array([[1.0], [5.0], [-2.0], [0.0]], dtype=np.float32)
    base_env = _DummyVectorEnv(rewards=rewards)
    env = RecordEpisodeStatistics(base_env)
    env.reset(seed=123)

    raw_return = np.zeros(1, dtype=np.float32)
    terminated = np.zeros(1, dtype=bool)
    truncated = np.zeros(1, dtype=bool)
    while not terminated.any() and not truncated.any():
        obs, reward, terminated, truncated, info = env.step(actions=np.zeros(1, dtype=np.int64))
        raw_return += reward

    assert "episode" in info
    assert np.allclose(info["episode"]["r"], raw_return)


def test_transform_reward_clips_rewards():
    raw_rewards = np.array([-3.0, 0.0, 2.5], dtype=np.float32)
    env = TransformReward(_DummyRecvEnv(raw_rewards), transform_fn=np.sign)
    obs, rewards, _, _, _ = env.recv()
    assert np.allclose(rewards, np.sign(raw_rewards))


if __name__ == "__main__":
    test_atari_vector_env_wrapper()
    test_record_episode_statistics_with_clipped_vs_raw_rewards()
    test_transform_reward_clips_rewards()
    test_torch_ops_wrapper()
    test_torch_ops_wrapper_episode_stats()
