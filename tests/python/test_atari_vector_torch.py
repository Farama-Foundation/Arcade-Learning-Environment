"""Tests for PyTorch custom ops (ale::send, ale::recv) in AtariVectorEnv."""

import numpy as np
import pytest
from ale_py import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence

torch = pytest.importorskip("torch")


def assert_rollout_equivalence(
    num_envs: int = 4,
    seeds: np.ndarray = np.arange(4),
    game: str = "pong",
    rollout_length: int = 100,
    **kwargs,
):
    """Compare plain env.step against ale::send + ale::recv over a rollout."""
    envs_1 = AtariVectorEnv(game, num_envs=num_envs, **kwargs)
    envs_2 = AtariVectorEnv(game, num_envs=num_envs, **kwargs)
    handle_id, ale_send, _, ale_recv, _, unregister = envs_2.torch()

    obs_1, info_1 = envs_1.reset(seed=seeds)
    obs_2, info_2 = envs_2.reset(seed=seeds)

    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)

    for _ in range(rollout_length):
        actions = envs_1.action_space.sample()
        actions_t = torch.as_tensor(actions, dtype=torch.int64)

        obs_1, rewards_1, terms_1, truncs_1, info_1 = envs_1.step(actions)

        ale_send(handle_id, actions_t)
        obs_2, rewards_2, terms_2, truncs_2, _ = ale_recv(handle_id)

        assert data_equivalence(obs_1, obs_2.numpy())
        assert data_equivalence(rewards_1, rewards_2.numpy())
        assert data_equivalence(terms_1, terms_2.numpy())
        assert data_equivalence(truncs_1, truncs_2.numpy())

    envs_1.close()
    unregister()
    envs_2.close()


@pytest.mark.parametrize(
    "num_envs, seeds",
    [(1, np.array([0])), (3, np.array([1, 2, 3])), (10, np.arange(10))],
)
def test_seeding(num_envs, seeds):
    """Rollout equivalence across num_envs and seed configurations."""
    assert_rollout_equivalence(num_envs=num_envs, seeds=seeds)


@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
@pytest.mark.parametrize("frame_skip", [1, 4])
@pytest.mark.parametrize("grayscale", [False, True])
def test_obs_params(stack_num, img_height, img_width, frame_skip, grayscale):
    """Rollout equivalence across observation shape parameters."""
    assert_rollout_equivalence(
        stack_num=stack_num,
        img_height=img_height,
        img_width=img_width,
        frameskip=frame_skip,
        grayscale=grayscale,
    )


def test_compile(
    num_envs: int = 4,
    seeds: np.ndarray = np.arange(4),
    game: str = "pong",
    rollout_length: int = 10,
):
    """torch.compile(fullgraph=True) traces through send+recv without graph breaks."""
    envs_1 = AtariVectorEnv(game, num_envs=num_envs)
    envs_2 = AtariVectorEnv(game, num_envs=num_envs)
    handle_id, ale_send, _, ale_recv, _, unregister = envs_2.torch()

    envs_1.reset(seed=seeds)
    envs_2.reset(seed=seeds)

    def step_fn(handle_id, actions):
        ale_send(handle_id, actions)
        obs, rewards, terms, truncs, _ = ale_recv(handle_id)
        return obs, rewards, terms, truncs

    compiled_step = torch.compile(step_fn, fullgraph=True)

    for _ in range(rollout_length):
        actions = envs_1.action_space.sample()
        actions_t = torch.as_tensor(actions, dtype=torch.int64)

        obs_1, rewards_1, terms_1, truncs_1, _ = envs_1.step(actions)
        obs_2, rewards_2, terms_2, truncs_2 = compiled_step(handle_id, actions_t)

        assert data_equivalence(obs_1, obs_2.numpy())
        assert data_equivalence(rewards_1, rewards_2.numpy())
        assert data_equivalence(terms_1, terms_2.numpy())
        assert data_equivalence(truncs_1, truncs_2.numpy())

    envs_1.close()
    unregister()
    envs_2.close()


def test_recv_pinned_memory():
    """All tensors returned by ale::recv are CPU-pinned for async GPU transfers."""
    env = AtariVectorEnv("pong", num_envs=2)
    handle_id, ale_send, _, ale_recv, _, unregister = env.torch()
    env.reset()

    ale_send(handle_id, torch.zeros(2, dtype=torch.int64))
    obs, rewards, terms, truncs, steps = ale_recv(handle_id)

    assert obs.is_pinned()
    assert rewards.is_pinned()
    assert terms.is_pinned()
    assert truncs.is_pinned()
    assert steps.is_pinned()

    unregister()
    env.close()


def test_unregister_cleans_up():
    """unregister() removes the handle from all internal registry dicts."""
    from ale_py._torch_ops import _torch_buffers, _torch_envs, _torch_last_info

    env = AtariVectorEnv("pong", num_envs=2)
    handle_id, _, _, _, _, unregister = env.torch()

    assert handle_id in _torch_envs
    assert handle_id in _torch_buffers

    unregister()
    env.close()

    assert handle_id not in _torch_envs
    assert handle_id not in _torch_buffers
    assert handle_id not in _torch_last_info


def test_get_last_info():
    """get_last_info() returns empty dict before recv and populated dict after."""
    env = AtariVectorEnv("pong", num_envs=2)
    handle_id, ale_send, _, ale_recv, get_last_info, unregister = env.torch()
    env.reset()

    assert get_last_info() == {}

    ale_send(handle_id, torch.zeros(2, dtype=torch.int64))
    ale_recv(handle_id)
    info = get_last_info()

    assert "env_id" in info
    assert "lives" in info

    unregister()
    env.close()


def test_send_sequences():
    """ale::send_sequences matches numpy list path for variable-length action sequences."""
    num_envs = 3
    seeds = np.arange(num_envs)

    envs_1 = AtariVectorEnv("pong", num_envs=num_envs)
    envs_2 = AtariVectorEnv("pong", num_envs=num_envs)
    handle_id, _, _, ale_recv, _, unregister = envs_2.torch()

    envs_1.reset(seed=seeds)
    envs_2.reset(seed=seeds)

    # [1, 2, 1] steps per env - variable length
    option_lengths = [1, 2, 1]
    action_sequences = [np.zeros(k, dtype=np.int64) for k in option_lengths]
    gamma = 0.99

    offsets = torch.tensor([0, 1, 3, 4], dtype=torch.int64)
    values = torch.zeros(sum(option_lengths), dtype=torch.int64)

    envs_1.send(action_sequences, gamma=gamma)
    obs_1, rewards_1, terms_1, truncs_1, _ = envs_1.recv()

    torch.ops.ale.send_sequences(handle_id, values, offsets, gamma)
    obs_2, rewards_2, terms_2, truncs_2, _ = ale_recv(handle_id)

    assert data_equivalence(obs_1, obs_2.numpy())
    assert data_equivalence(rewards_1, rewards_2.numpy())
    assert data_equivalence(terms_1, terms_2.numpy())
    assert data_equivalence(truncs_1, truncs_2.numpy())

    envs_1.close()
    unregister()
    envs_2.close()
