"""Wrappers for AtariVectorEnv: episode statistics, reward transform, and torch ops."""

from __future__ import annotations

import collections
import time
from typing import Union

import numpy as np
from gymnasium.logger import warn
from gymnasium.vector.vector_env import AutoresetMode

from ale_py.vector_env import AtariVectorEnv

try:
    from gymnasium.vector import (
        VectorEnv as _GymVectorEnv,
        VectorWrapper as _GymVectorWrapper,
        VectorRewardWrapper as _GymRewardWrapper,
        VectorObservationWrapper as _GymObsWrapper,
        VectorActionWrapper as _GymActionWrapper,
    )
    from gymnasium.wrappers.vector.common import (
        RecordEpisodeStatistics as _GymRecordEpisodeStats,
    )

    class _GymProxy(_GymVectorEnv):
        """Bare VectorEnv subclass used to satisfy isinstance checks when constructing
        Gymnasium wrappers purely for their transform methods (rewards/observations)."""

    _GYM_WRAPPERS_AVAILABLE = True
except ImportError:
    _GYM_WRAPPERS_AVAILABLE = False


def _make_gym_proxy(env):
    """Create a minimal VectorEnv proxy that passes Gymnasium isinstance checks."""
    proxy = object.__new__(_GymProxy)
    for attr in (
        "metadata",
        "action_space",
        "observation_space",
        "single_action_space",
        "single_observation_space",
        "num_envs",
    ):
        val = getattr(env, attr, None)
        if val is not None:
            setattr(proxy, attr, val)
    if not hasattr(proxy, "metadata"):
        proxy.metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}
    return proxy


def _apply_wrapper(env, cls, args):
    """Apply cls(env, *args), adapting Gymnasium vector wrappers to support recv()."""
    if not _GYM_WRAPPERS_AVAILABLE or not (
        isinstance(cls, type) and issubclass(cls, _GymVectorWrapper)
    ):
        return cls(env, *args)

    if cls is _GymRecordEpisodeStats:
        warn(
            f"{cls.__name__} does not implement recv() and was automatically replaced with "
            "ale_py.wrappers.RecordEpisodeStatistics. "
            "Pass ale_py.wrappers.RecordEpisodeStatistics explicitly to suppress this warning."
        )
        return RecordEpisodeStatistics(env, *args)

    if issubclass(cls, _GymRewardWrapper):
        if cls.rewards is _GymRewardWrapper.rewards:
            raise TypeError(
                f"{cls.__name__} subclasses VectorRewardWrapper but does not implement rewards(). "
                "Cannot be used with the recv() API."
            )
        gym_instance = cls(_make_gym_proxy(env), *args)

        class _GymRewardAdapter(AtariVectorEnvWrapper):
            def step(self, actions, gamma=1.0, paddle_strength=1.0):
                obs, reward, term, trunc, info = super().step(
                    actions, gamma=gamma, paddle_strength=paddle_strength
                )
                return obs, gym_instance.rewards(reward), term, trunc, info

            def recv(self):
                obs, reward, term, trunc, info = super().recv()
                return obs, gym_instance.rewards(reward), term, trunc, info

        _GymRewardAdapter.__name__ = f"{cls.__name__}Adapter"
        _GymRewardAdapter.__qualname__ = f"{cls.__name__}Adapter"
        return _GymRewardAdapter(env)

    if issubclass(cls, _GymObsWrapper):
        if cls.observations is _GymObsWrapper.observations:
            raise TypeError(
                f"{cls.__name__} subclasses VectorObservationWrapper but does not implement "
                "observations(). Cannot be used with the recv() API."
            )
        gym_instance = cls(_make_gym_proxy(env), *args)

        class _GymObsAdapter(AtariVectorEnvWrapper):
            def step(self, actions, gamma=1.0, paddle_strength=1.0):
                obs, reward, term, trunc, info = super().step(
                    actions, gamma=gamma, paddle_strength=paddle_strength
                )
                return gym_instance.observations(obs), reward, term, trunc, info

            def recv(self):
                obs, reward, term, trunc, info = super().recv()
                return gym_instance.observations(obs), reward, term, trunc, info

        _GymObsAdapter.__name__ = f"{cls.__name__}Adapter"
        _GymObsAdapter.__qualname__ = f"{cls.__name__}Adapter"
        return _GymObsAdapter(env)

    if issubclass(cls, _GymActionWrapper):
        if cls.actions is _GymActionWrapper.actions:
            raise TypeError(
                f"{cls.__name__} subclasses VectorActionWrapper but does not implement actions(). "
                "Cannot be adapted."
            )
        gym_instance = cls(_make_gym_proxy(env), *args)

        def _transform_flat(actions):
            return np.asarray(
                gym_instance.actions(np.asarray(actions, dtype=np.int64)), dtype=np.int64
            )

        def _transform_sequences(sequences):
            # Each element is a variable-length action array for one ROM.
            # The transform must be applied to each individual action within each sequence.
            result = []
            for seq in sequences:
                arr = np.asarray(seq, dtype=np.int64)
                result.append(
                    np.asarray(gym_instance.actions(arr), dtype=np.int64) if len(arr) > 0 else arr
                )
            return result

        def _is_sequences(actions):
            return isinstance(actions, (list, tuple)) and (
                len(actions) == 0 or not isinstance(actions[0], (int, np.integer))
            )

        class _GymActionAdapter(AtariVectorEnvWrapper):
            def send(self, actions, gamma=1.0, paddle_strength=1.0):
                actions = _transform_sequences(actions) if _is_sequences(actions) else _transform_flat(actions)
                super().send(actions, gamma=gamma, paddle_strength=paddle_strength)

            def step(self, actions, gamma=1.0, paddle_strength=1.0):
                actions = _transform_sequences(actions) if _is_sequences(actions) else _transform_flat(actions)
                return super().step(actions, gamma=gamma, paddle_strength=paddle_strength)

            def _flat_action_transform(self, actions):
                return _transform_flat(actions)

        _GymActionAdapter.__name__ = f"{cls.__name__}Adapter"
        _GymActionAdapter.__qualname__ = f"{cls.__name__}Adapter"
        return _GymActionAdapter(env)

    raise TypeError(
        f"{cls.__name__} is a Gymnasium VectorWrapper but does not subclass "
        "VectorRewardWrapper, VectorObservationWrapper, or VectorActionWrapper, "
        "so it cannot be adapted to support recv(). "
        "Use an ale_py.wrappers equivalent or add a recv() override."
    )


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

    def wrap(self, *wrappers):
        """Apply additional wrappers in sequence and return the outermost wrapper."""
        env = self
        for w in wrappers:
            cls, *args = w if isinstance(w, tuple) else (w,)
            env = _apply_wrapper(env, cls, args)
        return env

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
            _raw_ale_send,
            self.ale_step,
            _,
            self._unregister,
        ) = env.torch()

        flat_transforms = []
        e = env
        while hasattr(e, "env"):
            cls = type(e)
            if "_flat_action_transform" in cls.__dict__:
                flat_transforms.append(e._flat_action_transform)
            elif "step" in cls.__dict__ and "recv" not in cls.__dict__:
                warn(
                    f"{cls.__name__} overrides step() but not recv(). "
                    "Its processing will be skipped during ale_recv; "
                    "add a recv() override to support the async send/recv API."
                )
            e = e.env

        if flat_transforms:
            import torch as _torch

            def _ale_send_wrapped(handle_id, actions):
                arr = actions.cpu().numpy() if hasattr(actions, "cpu") else np.asarray(actions, dtype=np.int64)
                for transform in flat_transforms:
                    arr = transform(arr)
                _raw_ale_send(handle_id, _torch.as_tensor(arr, dtype=_torch.int64))

            self.ale_send = _ale_send_wrapped
        else:
            self.ale_send = _raw_ale_send

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

class _DummySendEnv:
    """Records the last send() call for verifying action transforms."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.last_actions = None
        self.metadata = {"autoreset_mode": AutoresetMode.SAME_STEP}

    def send(self, actions, gamma=1.0, paddle_strength=1.0):
        self.last_actions = actions

    def step(self, actions, gamma=1.0, paddle_strength=1.0):
        self.last_actions = actions
        obs = np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)
        return obs, np.zeros(self.num_envs, dtype=np.float32), np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), {}

    def recv(self):
        obs = np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)
        return obs, np.zeros(self.num_envs, dtype=np.float32), np.zeros(self.num_envs, dtype=bool), np.zeros(self.num_envs, dtype=bool), {}


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


def test_wrap_chain_structure():
    """wrap() must apply wrappers left-to-right."""
    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep", reward_clipping=False)
    wrapped = env.wrap(
        RecordEpisodeStatistics,
        (TransformReward, np.sign),
        TorchOpsWrapper,
    )
    assert isinstance(wrapped, TorchOpsWrapper)
    assert isinstance(wrapped.env, TransformReward)
    assert isinstance(wrapped.env.env, RecordEpisodeStatistics)
    assert isinstance(wrapped.env.env.env, AtariVectorEnv)
    wrapped.reset()
    wrapped.close()


def test_wrap_transform_reward_in_recv_chain():
    """wrap() must put TransformReward in the recv() path, not just step()."""
    # Return rewards outside [-1, 1] so sign-clipping is observable.
    raw_rewards = np.array([5.0, -3.0, 0.0], dtype=np.float32)
    base = _DummyRecvEnv(raw_rewards)
    wrapped = AtariVectorEnvWrapper(base).wrap((TransformReward, np.sign))
    _, rewards, _, _, _ = wrapped.recv()
    assert np.allclose(rewards, np.sign(raw_rewards)), f"TransformReward not applied: {rewards}"


def test_wrap_record_episode_statistics_in_recv_chain():
    """wrap() must put RecordEpisodeStatistics in the recv() path."""
    import torch

    wrapped = AtariVectorEnv(
        game="pong", num_envs=1, autoreset_mode="SameStep", reward_clipping=False
    ).wrap(RecordEpisodeStatistics, TorchOpsWrapper)

    assert isinstance(wrapped, TorchOpsWrapper)
    assert isinstance(wrapped.env, RecordEpisodeStatistics)

    wrapped.reset()
    actions = torch.zeros(1, dtype=torch.int64)
    episode_completed = False
    for _ in range(10000):
        wrapped.ale_send(wrapped.handle_id, actions)
        _, _, _, _, infos = wrapped.ale_recv(wrapped.handle_id)
        if "episode" in infos:
            assert "r" in infos["episode"]
            assert "l" in infos["episode"]
            episode_completed = True
            break
    assert episode_completed, "No episode completed in 10000 steps"
    wrapped.close()
    from ale_py._torch_ops import _torch_envs
    assert wrapped.handle_id not in _torch_envs


def test_wrap_on_wrapper():
    """wrap() on AtariVectorEnvWrapper must also chain correctly."""
    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep")
    wrapped = RecordEpisodeStatistics(env).wrap(TorchOpsWrapper)
    assert isinstance(wrapped, TorchOpsWrapper)
    assert isinstance(wrapped.env, RecordEpisodeStatistics)
    wrapped.reset()
    wrapped.close()


def test_gymnasium_transform_reward_adapter():
    """gymnasium.wrappers.vector.TransformReward must apply rewards() in recv()."""
    import warnings
    from gymnasium.wrappers.vector import TransformReward as GymTransformReward

    raw_rewards = np.array([5.0, -3.0, 0.0], dtype=np.float32)
    base = _DummyRecvEnv(raw_rewards)
    wrapped = AtariVectorEnvWrapper(base).wrap((GymTransformReward, np.sign))
    _, rewards, _, _, _ = wrapped.recv()
    assert np.allclose(rewards, np.sign(raw_rewards)), f"Gymnasium TransformReward not applied in recv(): {rewards}"


def test_gymnasium_record_episode_stats_swapped_with_warning():
    """gymnasium RecordEpisodeStatistics must be swapped for ale_py version with a warning."""
    import warnings
    from ale_py.wrappers import RecordEpisodeStatistics as AleRecordEpisodeStats
    from gymnasium.wrappers.vector import RecordEpisodeStatistics as GymRecordEpisodeStats

    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wrapped = env.wrap(GymRecordEpisodeStats)

    assert isinstance(wrapped, AleRecordEpisodeStats), (
        f"Expected ale_py RecordEpisodeStatistics, got {type(wrapped)}"
    )
    assert any(
        "replaced" in str(w.message).lower() or "RecordEpisodeStatistics" in str(w.message)
        for w in caught
    ), "Expected a warning about automatic replacement"
    wrapped.reset()
    wrapped.close()


def test_gymnasium_action_adapter_flat():
    """gymnasium VectorActionWrapper must transform flat actions in send() and step()."""
    from gymnasium.vector import VectorActionWrapper as GymActionWrapper

    class _AddOneAction(GymActionWrapper):
        def actions(self, actions):
            return (np.asarray(actions) + 1) % 4

    dummy = _DummySendEnv(num_envs=2)
    wrapped = AtariVectorEnvWrapper(dummy).wrap(_AddOneAction)

    wrapped.send(np.array([0, 2], dtype=np.int64))
    assert np.array_equal(dummy.last_actions, np.array([1, 3], dtype=np.int64)), (
        f"send() flat transform failed: {dummy.last_actions}"
    )

    wrapped.step(np.array([3, 1], dtype=np.int64))
    assert np.array_equal(dummy.last_actions, np.array([0, 2], dtype=np.int64)), (
        f"step() flat transform failed: {dummy.last_actions}"
    )


def test_gymnasium_action_adapter_sequences():
    """gymnasium VectorActionWrapper must transform each action within each ROM's sequence."""
    from gymnasium.vector import VectorActionWrapper as GymActionWrapper

    class _AddOneAction(GymActionWrapper):
        def actions(self, actions):
            return (np.asarray(actions) + 1) % 4

    dummy = _DummySendEnv(num_envs=2)
    wrapped = AtariVectorEnvWrapper(dummy).wrap(_AddOneAction)

    sequences = [np.array([0, 2, 1], dtype=np.int64), np.array([3], dtype=np.int64)]
    wrapped.send(sequences)

    transformed = dummy.last_actions
    assert transformed is not None
    assert np.array_equal(transformed[0], np.array([1, 3, 2])), (
        f"ROM 0 sequence transform failed: {transformed[0]}"
    )
    assert np.array_equal(transformed[1], np.array([0])), (
        f"ROM 1 sequence transform failed: {transformed[1]}"
    )


def test_gymnasium_action_adapter_empty_sequence():
    """Empty sequence (wait action) must pass through unchanged."""
    from gymnasium.vector import VectorActionWrapper as GymActionWrapper

    class _AddOneAction(GymActionWrapper):
        def actions(self, actions):
            return (np.asarray(actions) + 1) % 4

    dummy = _DummySendEnv(num_envs=2)
    wrapped = AtariVectorEnvWrapper(dummy).wrap(_AddOneAction)

    sequences = [np.array([0], dtype=np.int64), np.array([], dtype=np.int64)]
    wrapped.send(sequences)
    last = dummy.last_actions
    assert last is not None
    assert len(last[1]) == 0, "Empty sequence must stay empty"
    assert np.array_equal(last[0], np.array([1])), (
        f"Non-empty ROM transform failed alongside empty: {last[0]}"
    )


def test_gymnasium_action_adapter_ale_send():
    """TorchOpsWrapper must apply action transforms when ale_send is called."""
    import torch
    from gymnasium.vector import VectorActionWrapper as GymActionWrapper

    class _ZeroAction(GymActionWrapper):
        """Always sends action 0 regardless of input."""
        def actions(self, actions):
            return np.zeros_like(np.asarray(actions))

    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep")
    wrapped = env.wrap(_ZeroAction, TorchOpsWrapper)

    wrapped.reset()
    # ale_send receives action 5 but _ZeroAction remaps to 0 - env should not crash
    wrapped.ale_send(wrapped.handle_id, torch.tensor([5], dtype=torch.int64))
    obs, *_ = wrapped.ale_recv(wrapped.handle_id)
    assert obs.shape == (1, 4, 84, 84)
    wrapped.close()


def test_gymnasium_plain_wrapper_raises():
    """A plain Gymnasium VectorWrapper with no rewards()/observations() must raise TypeError."""
    import pytest
    from gymnasium.vector import VectorWrapper as GymVectorWrapper

    class _PlainWrapper(GymVectorWrapper):
        pass

    env = AtariVectorEnv(game="pong", num_envs=1, autoreset_mode="SameStep")
    try:
        env.wrap(_PlainWrapper)
        assert False, "Expected TypeError"
    except TypeError as e:
        assert "recv" in str(e).lower() or "VectorWrapper" in str(e)
    finally:
        env.close()


if __name__ == "__main__":
    test_atari_vector_env_wrapper()
    test_record_episode_statistics_with_clipped_vs_raw_rewards()
    test_transform_reward_clips_rewards()
    test_torch_ops_wrapper()
    test_torch_ops_wrapper_episode_stats()
    test_wrap_chain_structure()
    test_wrap_transform_reward_in_recv_chain()
    test_wrap_record_episode_statistics_in_recv_chain()
    test_wrap_on_wrapper()
    test_gymnasium_transform_reward_adapter()
    test_gymnasium_record_episode_stats_swapped_with_warning()
    test_gymnasium_action_adapter_flat()
    test_gymnasium_action_adapter_sequences()
    test_gymnasium_action_adapter_empty_sequence()
    test_gymnasium_action_adapter_ale_send()
    test_gymnasium_plain_wrapper_raises()
