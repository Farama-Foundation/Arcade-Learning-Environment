"""Test equivalence between Gymnasium and Vector environments."""

import gymnasium as gym
import numpy as np
import pytest
from ale_py.vector_env import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
@pytest.mark.parametrize("grayscale", [True, False])
def test_reset_step_shapes(num_envs, stack_num, img_height, img_width, grayscale):
    """Test if reset returns observations with the correct shape."""
    envs = AtariVectorEnv(
        game="breakout",
        num_envs=num_envs,
        stack_num=stack_num,
        img_height=img_height,
        img_width=img_width,
        grayscale=grayscale,
    )

    expected_shape = (num_envs, stack_num, img_height, img_width)
    if not grayscale:
        expected_shape += (3,)

    assert envs.num_envs == num_envs
    assert envs.observation_space.shape == expected_shape
    assert envs.action_space.shape == (num_envs,)

    obs, info = envs.reset(seed=0)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == expected_shape
    assert obs.dtype == np.uint8
    assert obs in envs.observation_space, f"{envs.observation_space=}"
    assert isinstance(info, dict)
    assert all(
        isinstance(val, np.ndarray) and len(val) == num_envs for val in info.values()
    )

    actions = envs.action_space.sample()
    obs, reward, terminations, truncations, info = envs.step(actions)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == expected_shape
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


NUM_ENVS = 8


def assert_rollout_equivalence(
    gym_envs,
    ale_envs,
    rollout_length=100,
    reset_seed=123,
    action_seed=123,
):
    """Test if both environments produce similar results over a short rollout."""
    assert gym_envs.num_envs == ale_envs.num_envs
    assert gym_envs.observation_space == ale_envs.observation_space
    assert gym_envs.action_space == ale_envs.action_space

    gym_obs, gym_info = gym_envs.reset(seed=reset_seed)
    ale_obs, ale_info = ale_envs.reset(seed=reset_seed)

    assert data_equivalence(gym_obs, ale_obs)

    gym_info = {
        key: value.astype(np.int32)
        for key, value in gym_info.items()
        if not key.startswith("_") and key != "seeds"
    }
    env_ids = ale_info.pop("env_id")
    assert np.all(env_ids == np.arange(gym_envs.num_envs))
    assert data_equivalence(gym_info, ale_info)

    ale_envs.action_space.seed(action_seed)
    for i in range(rollout_length):
        actions = ale_envs.action_space.sample()

        gym_obs, gym_rewards, gym_terminations, gym_truncations, gym_info = (
            gym_envs.step(actions)
        )
        ale_obs, ale_rewards, ale_terminations, ale_truncations, ale_info = (
            ale_envs.step(actions)
        )

        if not data_equivalence(gym_obs, ale_obs):
            # For MacOS ARM, there is a known problem where there is a max difference of 1 for 1 or 2 pixels
            diff = gym_obs.astype(np.int32) - ale_obs.astype(np.int32)
            gym.logger.warn(
                f"rollout obs diff for timestep={i}, max diff={np.max(diff)}, min diff={np.min(diff)}, non-zero count={np.count_nonzero(diff)}"
            )

        assert data_equivalence(gym_rewards.astype(np.int32), ale_rewards)
        assert data_equivalence(gym_terminations, ale_terminations)
        assert data_equivalence(gym_truncations, ale_truncations)

        gym_info = {
            key: value.astype(np.int32)
            for key, value in gym_info.items()
            if not key.startswith("_") and key != "seeds"
        }
        env_ids = ale_info.pop("env_id")
        assert np.all(env_ids == np.arange(gym_envs.num_envs))
        assert data_equivalence(gym_info, ale_info)

    gym_envs.close()
    ale_envs.close()


@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
@pytest.mark.parametrize("frame_skip", [1, 4])
@pytest.mark.parametrize("grayscale", [False, True])
def test_obs_params(stack_num, img_height, img_width, frame_skip, grayscale):
    gym_envs = gym.vector.SyncVectorEnv(
        [
            lambda: gym.wrappers.FrameStackObservation(
                gym.wrappers.AtariPreprocessing(
                    gym.make("BreakoutNoFrameskip-v4"),
                    noop_max=0,
                    frame_skip=frame_skip,
                    screen_size=(img_width, img_height),
                    grayscale_obs=grayscale,
                ),
                stack_size=stack_num,
                padding_type="zero",
            )
            for _ in range(NUM_ENVS)
        ],
    )
    ale_envs = AtariVectorEnv(
        game="breakout",
        num_envs=NUM_ENVS,
        frameskip=frame_skip,
        img_height=img_height,
        img_width=img_width,
        stack_num=stack_num,
        noop_max=0,
        use_fire_reset=False,
        maxpool=frame_skip > 1,
        grayscale=grayscale,
    )

    assert_rollout_equivalence(gym_envs, ale_envs)


@pytest.mark.parametrize(
    "num_envs, seeds",
    [(1, np.array([0])), (3, np.array([1, 2, 3])), (10, np.arange(10))],
)
@pytest.mark.parametrize("noop_max", (0, 10, 30))
@pytest.mark.parametrize("repeat_action_probability", (0.0, 0.25))
@pytest.mark.parametrize("use_fire_reset", [False, True])
def test_determinism(
    num_envs: int,
    seeds: np.ndarray,
    noop_max: int,
    repeat_action_probability: float,
    use_fire_reset: bool,
    game: str = "pong",
    rollout_length: int = 100,
):
    envs_1 = AtariVectorEnv(
        game,
        num_envs=num_envs,
        noop_max=noop_max,
        repeat_action_probability=repeat_action_probability,
        use_fire_reset=use_fire_reset,
    )
    envs_2 = AtariVectorEnv(
        game,
        num_envs=num_envs,
        noop_max=noop_max,
        repeat_action_probability=repeat_action_probability,
        use_fire_reset=use_fire_reset,
    )

    obs_1, info_1 = envs_1.reset(seed=seeds)
    obs_2, info_2 = envs_2.reset(seed=seeds)

    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)

    for i in range(rollout_length):
        actions = envs_1.action_space.sample()

        obs_1, rewards_1, terminations_1, truncations_1, info_1 = envs_1.step(actions)
        obs_2, rewards_2, terminations_2, truncations_2, info_2 = envs_2.step(actions)

        assert data_equivalence(obs_1, obs_2)
        assert data_equivalence(rewards_1, rewards_2)
        assert data_equivalence(terminations_1, terminations_2)
        assert data_equivalence(truncations_1, truncations_2)
        assert data_equivalence(info_1, info_2)

    envs_1.close()
    envs_2.close()


def test_batch_size_async():
    pass  # TODO


def test_episodic_life_and_life_loss_info():
    pass  # TODO
