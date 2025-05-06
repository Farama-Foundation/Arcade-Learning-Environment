import numpy as np
from ale_py import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence


def test_vector_xla_equivalence(
    game: str = "pong",
    num_envs: int = 3,
    seeds: np.ndarray = np.arange(3),
    rollout_length: int = 100,
):
    envs_1 = AtariVectorEnv(game, num_envs=num_envs)
    envs_2 = AtariVectorEnv(game, num_envs=num_envs)

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
