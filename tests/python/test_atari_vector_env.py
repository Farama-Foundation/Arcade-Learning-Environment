"""Test equivalence between Gymnasium and Vector environments."""

import gymnasium as gym
import numpy as np
import pytest
from ale_py.vector_env import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
def test_reset_step_shapes(num_envs, stack_num, img_height, img_width):
    """Test if reset returns observations with the correct shape."""
    envs = AtariVectorEnv(
        game="breakout",
        num_envs=num_envs,
        stack_num=stack_num,
        img_height=img_height,
        img_width=img_width,
    )

    assert envs.num_envs == num_envs
    assert envs.observation_space.shape == (num_envs, stack_num, img_height, img_width)
    assert envs.action_space.shape == (num_envs,)

    obs, info = envs.reset(seed=0)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (num_envs, stack_num, img_height, img_width)
    assert obs.dtype == np.uint8
    assert obs in envs.observation_space, f"{envs.observation_space=}"
    assert isinstance(info, dict)
    assert all(
        isinstance(val, np.ndarray) and len(val) == num_envs for val in info.values()
    )

    actions = envs.action_space.sample()
    obs, reward, terminations, truncations, info = envs.step(actions)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == (num_envs, stack_num, img_height, img_width)
    assert obs.dtype == np.uint8
    assert obs in envs.observation_space, f"{envs.observation_space=}"
    assert isinstance(reward, np.ndarray) and reward.dtype == np.int32
    assert reward.shape == (num_envs,)
    assert isinstance(terminations, np.ndarray) and terminations.dtype == bool
    assert terminations.shape == (num_envs,)
    assert isinstance(truncations, np.ndarray) and truncations.dtype == bool
    assert truncations.shape == (num_envs,)
    assert isinstance(info, dict)
    assert all(
        isinstance(val, np.ndarray) and len(val) == num_envs for val in info.values()
    )

    envs.close()


@pytest.mark.parametrize("num_envs", [1, 8])
@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
@pytest.mark.parametrize("frame_skip", [1, 4])
def test_rollout_consistency(
    num_envs, stack_num, img_height, img_width, frame_skip, rollout_length=100
):
    """Test if both environments produce similar results over a short rollout."""
    gym_envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.wrappers.FrameStackObservation(
                gym.wrappers.AtariPreprocessing(
                    gym.make("BreakoutNoFrameskip-v4"),
                    noop_max=0,
                    frame_skip=frame_skip,
                    screen_size=(img_width, img_height),
                ),
                stack_size=stack_num,
                padding_type="zero",
            )
            for _ in range(num_envs)
        ],
    )
    ale_envs = AtariVectorEnv(
        game="breakout",
        num_envs=num_envs,
        frameskip=frame_skip,
        img_height=img_height,
        img_width=img_width,
        stack_num=stack_num,
        noop_max=0,
        use_fire_reset=False,
        maxpool=frame_skip > 1,
    )

    assert gym_envs.num_envs == ale_envs.num_envs
    assert gym_envs.observation_space == ale_envs.observation_space
    assert gym_envs.action_space == ale_envs.action_space

    gym_obs, gym_info = gym_envs.reset(seed=123)
    ale_obs, ale_info = ale_envs.reset(seed=123)

    assert data_equivalence(gym_obs, ale_obs)

    gym_info = {
        key: value.astype(np.int32)
        for key, value in gym_info.items()
        if not key.startswith("_") and key != "seeds"
    }
    env_ids = ale_info.pop("env_id")
    assert np.all(env_ids == np.arange(num_envs))
    assert data_equivalence(gym_info, ale_info)

    ale_envs.action_space.seed(123)
    for i in range(rollout_length):
        actions = ale_envs.action_space.sample()

        gym_obs, gym_rewards, gym_terminations, gym_truncations, gym_info = (
            gym_envs.step(actions)
        )
        ale_obs, ale_rewards, ale_terminations, ale_truncations, ale_info = (
            ale_envs.step(actions)
        )

        assert data_equivalence(gym_obs, ale_obs), i
        assert data_equivalence(gym_rewards.astype(np.int32), ale_rewards)
        assert data_equivalence(gym_terminations, ale_terminations)
        assert data_equivalence(gym_truncations, ale_truncations)

        gym_info = {
            key: value.astype(np.int32)
            for key, value in gym_info.items()
            if not key.startswith("_") and key != "seeds"
        }
        env_ids = ale_info.pop("env_id")
        assert np.all(env_ids == np.arange(num_envs))
        assert data_equivalence(gym_info, ale_info)

    gym_envs.close()
    ale_envs.close()


def test_batch_size():
    pass  # TODO


def test_full_action_space():
    pass  # TODO


def test_max_episode_steps():
    pass  # TODO


def test_episodic_life_and_life_loss_info():
    pass  # TODO


def test_continuous_action_space():
    pass  # TODO
