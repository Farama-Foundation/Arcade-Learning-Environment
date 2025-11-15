import numpy as np
import pytest
from ale_py import AtariVectorEnv
from gymnasium.utils.env_checker import data_equivalence

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
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

    # Convert JAX arrays to numpy for comparison
    assert data_equivalence(obs_1, np.asarray(obs_2))
    assert data_equivalence(info_1, {k: np.asarray(v) for k, v in info_2.items()})

    for _ in range(rollout_length):
        actions = envs_1.action_space.sample()

        obs_1, rewards_1, terminations_1, truncations_1, info_1 = envs_1.step(actions)
        env_2_handle, (obs_2, rewards_2, terminations_2, truncations_2, info_2) = (
            env_2_step(env_2_handle, actions)
        )

        # Convert JAX arrays to numpy for comparison
        assert data_equivalence(obs_1, np.asarray(obs_2))
        assert data_equivalence(rewards_1, np.asarray(rewards_2))
        assert data_equivalence(terminations_1, np.asarray(terminations_2))
        assert data_equivalence(truncations_1, np.asarray(truncations_2))
        assert data_equivalence(info_1, {k: np.asarray(v) for k, v in info_2.items()})

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


@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8])
def test_continuous_actions(threshold):
    assert_rollout_equivalence(continuous=True, continuous_action_threshold=threshold)


@pytest.mark.parametrize("continuous", [True, False])
def test_jit(
    continuous: bool,
    num_envs: int = 4,
    seeds: np.ndarray = np.arange(4),
    game: str = "pong",
    rollout_length: int = 100,
    **kwargs,
):
    chex.clear_trace_counter()

    envs_1 = AtariVectorEnv(game, num_envs=num_envs, continuous=continuous, **kwargs)
    envs_2 = AtariVectorEnv(game, num_envs=num_envs, continuous=continuous, **kwargs)

    env_2_handle, env_2_reset, env_2_step = envs_2.xla()

    # Validate reset equivalence
    obs_1, info_1 = envs_1.reset(seed=seeds)
    env_2_handle, (obs_2, info_2) = env_2_reset(env_2_handle, seed=jnp.array(seeds))
    assert data_equivalence(obs_1, np.asarray(obs_2))
    assert data_equivalence(info_1, {k: np.asarray(v) for k, v in info_2.items()})

    # Actions for environment rollout
    rollout_actions = [envs_1.action_space.sample() for _ in range(rollout_length)]

    # Rollout for VectorAtariEnv
    env_rollout = [
        envs_1.step(rollout_actions[time_step]) for time_step in range(rollout_length)
    ]

    # Rollout for VectorAtariEnv XLA
    @jax.jit
    @chex.assert_max_traces(1)
    def actor(handle, action):
        return env_2_step(handle, action)

    _, xla_rollout = jax.lax.scan(actor, env_2_handle, xs=jnp.array(rollout_actions))
    obs_xla, rewards_xla, terms_xla, truncs_xla, info_xla = xla_rollout

    # Compare the env-rollouts and the xla-rollouts
    for i in range(rollout_length):
        obs_1, reward_1, term_1, trunc_1, info_1 = env_rollout[i]

        obs_2 = np.asarray(obs_xla[i])
        reward_2 = np.asarray(rewards_xla[i])
        term_2 = np.asarray(terms_xla[i])
        trunc_2 = np.asarray(truncs_xla[i])
        info_2 = {k: np.asarray(v[i]) for k, v in info_xla.items()}

        assert data_equivalence(obs_1, obs_2)
        assert data_equivalence(reward_1, reward_2)
        assert data_equivalence(term_1, term_2)
        assert data_equivalence(trunc_1, trunc_2)
        assert data_equivalence(info_1, info_2)


    envs_1.close()
    envs_2.close()
