import numpy as np
import pytest
from ale_py import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence

jax = pytest.importorskip("jax")
chex = pytest.importorskip("chex")


def assert_rollout_equivalence(
    num_envs: int = 4,
    seeds: np.ndarray = np.arange(4),
    game: str = "pong",
    rollout_length: int = 100,
    **kwargs,
):
    envs_1 = AtariVectorEnv(game, num_envs=num_envs, **kwargs)
    envs_2 = AtariVectorEnv(game, num_envs=num_envs, **kwargs)

    env_2_handle, env_2_reset, env_2_step = envs_2.xla()

    obs_1, info_1 = envs_1.reset(seed=seeds)
    env_2_handle, (obs_2, info_2) = env_2_reset(env_2_handle, seed=seeds)

    assert data_equivalence(obs_1, obs_2)
    assert data_equivalence(info_1, info_2)

    for _ in range(rollout_length):
        actions = envs_1.action_space.sample()

        obs_1, rewards_1, terminations_1, truncations_1, info_1 = envs_1.step(actions)
        env_2_handle, (obs_2, rewards_2, terminations_2, truncations_2, info_2) = (
            env_2_step(env_2_handle, actions)
        )

        assert data_equivalence(obs_1, obs_2)
        assert data_equivalence(rewards_1, rewards_2)
        assert data_equivalence(terminations_1, terminations_2)
        assert data_equivalence(truncations_1, truncations_2)
        assert data_equivalence(info_1, info_2)

    envs_1.close()
    envs_2.close()


@pytest.mark.parametrize(
    "num_envs, seeds",
    [(1, np.array([0])), (3, np.array([1, 2, 3])), (10, np.arange(10))],
)
def test_seeding(num_envs, seeds):
    assert_rollout_equivalence(num_envs, seeds)


@pytest.mark.parametrize("stack_num", [4, 6])
@pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
@pytest.mark.parametrize("frame_skip", [1, 4])
@pytest.mark.parametrize("grayscale", [False, True])
def test_obs_params(stack_num, img_height, img_width, frame_skip, grayscale):
    assert_rollout_equivalence(
        stack_num=stack_num,
        img_height=img_height,
        img_width=img_width,
        frameskip=frame_skip,
        grayscale=grayscale,
    )
