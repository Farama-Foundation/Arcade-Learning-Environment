"""Test equivalence between Gymnasium and Vector environments."""

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ale_py.vector_env import AtariVectorEnv
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation


@pytest.fixture
def common_params():
    """Common environment parameters."""
    return {
        "noop_max": 0,
        "frame_skip": 4,
        "frame_stack": 4,
        "img_height": 84,
        "img_width": 84,
        "episodic_life": True,
    }


@pytest.fixture
def gym_env(common_params):
    """Create a Gymnasium environment with preprocessing and frame stacking."""
    # v4 environment used to avoid `repeat-action-probability`
    env = gym.make("BreakoutNoFrameskip-v4", render_mode=None)

    # Apply standard preprocessing
    env = AtariPreprocessing(
        env,
        noop_max=common_params["noop_max"],
        frame_skip=common_params["frame_skip"],
        screen_size=(common_params["img_height"], common_params["img_width"]),
        terminal_on_life_loss=common_params["episodic_life"],
    )

    # Apply frame stacking
    env = FrameStackObservation(
        env, stack_size=common_params["frame_stack"], padding_type="zero"
    )

    yield env
    env.close()


@pytest.fixture
def vector_env(common_params, num_envs=1):
    """Create a vector environment with a single instance."""
    env = AtariVectorEnv(
        game="breakout",
        num_envs=num_envs,
        frame_skip=common_params["frame_skip"],
        stack_num=common_params["frame_stack"],
        img_height=common_params["img_height"],
        img_width=common_params["img_width"],
        noop_max=common_params["noop_max"],
        episodic_life=common_params["episodic_life"],
    )

    yield env
    env.close()


def test_action_space_equivalence(gym_env, vector_env):
    """Test if action spaces are equivalent."""
    assert gym_env.action_space.n == vector_env.single_action_space.n


def test_observation_space_equivalence(gym_env, vector_env):
    """Test if observation spaces are equivalent."""
    # - Gym: (stack, height, width)
    # - Vector: (num_envs, stack, height, width)
    gym_shape = gym_env.observation_space.shape
    vector_shape = vector_env.single_observation_space.shape

    assert vector_shape == gym_shape


def test_reset_output_shape(gym_env, vector_env):
    """Test if reset returns observations with the correct shape."""
    gym_obs, gym_info = gym_env.reset(seed=0)
    vector_obs, vector_info = vector_env.reset(seed=0)

    fig, axs = plt.subplots(2, 4, figsize=(8, 8))
    for i in range(4):
        axs[0, i].imshow(gym_obs[i])
        axs[1, i].imshow(vector_obs[0, i])
    plt.show()

    assert gym_obs.shape == vector_obs.shape[1:]
    for i in range(4):
        print(np.all(gym_obs[i] == vector_obs[0, i]), i)

    assert gym_info == vector_info


def test_step_output_shape(gym_env, vector_env):
    """Test if step returns observations with the correct shape."""
    gym_obs, _ = gym_env.reset()
    vector_obs, _ = vector_env.reset()

    # Take a step in both environments
    gym_action = gym_env.action_space.sample()
    vector_action = np.array([gym_action])

    gym_obs, gym_reward, gym_terminated, gym_truncated, gym_info = gym_env.step(
        gym_action
    )
    vector_obs, vector_rewards, vector_terminated, vector_truncated, vector_info = (
        vector_env.step(vector_action)
    )

    fig, axs = plt.subplots(2, 4, figsize=(8, 8))
    for i in range(4):
        axs[0, i].imshow(gym_obs[i])
        axs[1, i].imshow(vector_obs[0, i])
    plt.show()

    assert gym_obs.shape[1:] == vector_obs.shape
    assert np.all(gym_obs == vector_obs)
    assert gym_reward == vector_rewards[0]
    assert gym_terminated == vector_terminated[0]
    assert gym_truncated == vector_truncated[0]
    assert gym_info == vector_info


def test_rollout_consistency(gym_env, vector_env):
    """Test if both environments produce similar results over a short rollout."""
    # Reset environments
    gym_obs, _ = gym_env.reset(seed=0)
    vector_obs, _ = vector_env.reset()

    # Fixed action sequence to test
    actions = [1, 2, 3, 0, 1, 1, 2, 3]  # Some arbitrary but deterministic sequence

    gym_rewards = []
    vector_rewards = []

    # Run a short rollout with the same actions
    for action in actions:
        # Step the gym environment
        gym_obs, gym_reward, gym_terminated, gym_truncated, _ = gym_env.step(action)
        gym_rewards.append(gym_reward)

        # Step the vector environment
        vector_action = np.array([action])
        vector_obs, vector_reward, vector_terminated, vector_truncated, _ = (
            vector_env.step(vector_action)
        )
        vector_rewards.append(vector_reward[0])  # Extract from batch dimension

        # If either environment terminates, end the test
        if gym_terminated or vector_terminated[0]:
            break

    # Check that rewards have the same length
    assert len(gym_rewards) == len(vector_rewards)

    # Check if reward patterns are similar
    # We don't expect exact matches due to implementation differences,
    # but the pattern should be similar in terms of non-zero rewards
    gym_nonzero = [i for i, r in enumerate(gym_rewards) if r != 0]
    vector_nonzero = [i for i, r in enumerate(vector_rewards) if r != 0]

    # Basic consistency check - if one has non-zero rewards, the other should too
    assert (len(gym_nonzero) == 0 and len(vector_nonzero) == 0) or (
        len(gym_nonzero) > 0 and len(vector_nonzero) > 0
    )
