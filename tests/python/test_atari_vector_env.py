"""Test equivalence between Gymnasium and Vector environments."""

import ale_py
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import data_equivalence

gym.register_envs(ale_py)


def obs_equivalence(obs_1, obs_2, t, **log_kwargs):
    """Tests the equivalence between two observations.

    This is critical as we found that MacOS ARM and Python implementation had minor differences in output.
    Appearing when testing the Gymnasium and ALE vectorized environments.
    These differences are normally between 2 or 3 pixel of max difference of 1 or 2.

    As a result, we couldn't use `data_equivalence` and need this function.
    """
    assert obs_1.shape == obs_2.shape
    assert obs_1.dtype == obs_2.dtype

    diff = obs_1.astype(np.int32) - obs_2.astype(np.int32)
    count = np.count_nonzero(diff)
    if count > 1:
        assert obs_1.shape == obs_2.shape, t
        assert obs_1.dtype == obs_2.dtype, t
        assert (
            count <= 25
        ), f"timestep={t}, max diff={np.max(diff)}, min diff={np.min(diff)}, non-zero count={count}"
        assert (
            np.max(diff) <= 2
        ), f"timestep={t}, max diff={np.max(diff)}, min diff={np.min(diff)}, non-zero count={count}"
        assert (
            np.min(diff) >= -2
        ), f"timestep={t}, max diff={np.max(diff)}, min diff={np.min(diff)}, non-zero count={count}"

        gym.logger.warn(
            f"rollout obs diff - max diff={np.max(diff)}, min diff={np.min(diff)}, non-zero count={count}, params={log_kwargs}"
        )
    return True


def assert_gym_ale_rollout_equivalence(
    gym_envs, ale_envs, rollout_length=100, reset_seed=123, action_seed=123, **kwargs
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
    for t in range(rollout_length):
        actions = ale_envs.action_space.sample()

        gym_obs, gym_rewards, gym_terminations, gym_truncations, gym_info = (
            gym_envs.step(actions)
        )
        ale_obs, ale_rewards, ale_terminations, ale_truncations, ale_info = (
            ale_envs.step(actions)
        )

        assert obs_equivalence(gym_obs, ale_obs, t, **kwargs)
        assert data_equivalence(gym_rewards.astype(np.int32), ale_rewards), t
        assert data_equivalence(gym_terminations, ale_terminations), t
        assert data_equivalence(gym_truncations, ale_truncations), t

        gym_info = {
            key: value.astype(np.int32)
            for key, value in gym_info.items()
            if not key.startswith("_") and key != "seeds"
        }
        env_ids = ale_info.pop("env_id")
        assert np.all(env_ids == np.arange(gym_envs.num_envs)), t
        assert data_equivalence(gym_info, ale_info), t

    gym_envs.close()
    ale_envs.close()


@pytest.mark.parametrize("env_id", ["ALE/Breakout-v5"])
# @pytest.mark.parametrize(
#     "env_id", [env_id for env_id in gym.registry if "ALE/" in env_id]
# )
class TestVectorEnv:

    disable_vector_args = dict(
        noop_max=0,
        use_fire_reset=False,
        reward_clipping=False,
        repeat_action_probability=0.0,
    )
    disable_env_args = dict(frameskip=1, repeat_action_probability=0.0)
    disable_preprocessing_args = dict(noop_max=0)

    @pytest.mark.parametrize("num_envs", [1, 3])
    @pytest.mark.parametrize("stack_num", [4, 6])
    @pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
    @pytest.mark.parametrize("grayscale", [True, False])
    def test_reset_step_shapes(
        self, env_id, num_envs, stack_num, img_height, img_width, grayscale
    ):
        """Test if reset returns observations with the correct shape."""
        envs = gym.make_vec(
            env_id,
            num_envs,
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
            isinstance(val, np.ndarray) and len(val) == num_envs
            for val in info.values()
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
            isinstance(val, np.ndarray) and len(val) == num_envs
            for val in info.values()
        )

        envs.close()

    @pytest.mark.parametrize("stack_num", [4, 6])
    @pytest.mark.parametrize("img_height, img_width", [(84, 84), (210, 160)])
    @pytest.mark.parametrize("frame_skip", [1, 4])
    @pytest.mark.parametrize("grayscale", [False, True])
    def test_obs_params_equivalence(
        self,
        env_id,
        stack_num,
        img_height,
        img_width,
        frame_skip,
        grayscale,
        num_envs=8,
    ):
        gym_envs = gym.vector.SyncVectorEnv(
            [
                lambda: gym.wrappers.FrameStackObservation(
                    gym.wrappers.AtariPreprocessing(
                        gym.make(env_id, **self.disable_env_args),
                        frame_skip=frame_skip,
                        screen_size=(img_width, img_height),
                        grayscale_obs=grayscale,
                        **self.disable_preprocessing_args,
                    ),
                    stack_size=stack_num,
                    padding_type="zero",
                )
                for _ in range(num_envs)
            ],
        )
        ale_envs = gym.make_vec(
            env_id,
            num_envs,
            frameskip=frame_skip,
            img_height=img_height,
            img_width=img_width,
            stack_num=stack_num,
            maxpool=frame_skip > 1,
            grayscale=grayscale,
            **self.disable_vector_args,
        )

        assert_gym_ale_rollout_equivalence(
            gym_envs,
            ale_envs,
            stack_num=stack_num,
            img_height=img_height,
            img_width=img_width,
            frame_skip=frame_skip,
            grayscale=grayscale,
        )

    @pytest.mark.parametrize("continuous_action_threshold", (0.2, 0.5, 0.8))
    def test_continuous_equivalence(
        self, env_id, continuous_action_threshold, num_envs=8
    ):
        gym_envs = gym.vector.SyncVectorEnv(
            [
                lambda: gym.wrappers.FrameStackObservation(
                    gym.wrappers.AtariPreprocessing(
                        gym.make(
                            env_id,
                            continuous=True,
                            continuous_action_threshold=continuous_action_threshold,
                            **self.disable_env_args,
                        ),
                        **self.disable_preprocessing_args,
                    ),
                    stack_size=4,
                    padding_type="zero",
                )
                for _ in range(num_envs)
            ],
        )
        ale_envs = gym.make_vec(
            env_id,
            num_envs,
            continuous=True,
            continuous_action_threshold=continuous_action_threshold,
            **self.disable_vector_args,
        )

        assert_gym_ale_rollout_equivalence(
            gym_envs,
            ale_envs,
            continuous=True,
            continuous_action_threshold=continuous_action_threshold,
        )

    @pytest.mark.parametrize(
        "num_envs, seeds",
        [(1, np.array([0])), (3, np.array([1, 2, 3])), (10, np.arange(10))],
    )
    @pytest.mark.parametrize("noop_max", (0, 10, 30))
    @pytest.mark.parametrize("repeat_action_probability", (0.0, 0.25))
    @pytest.mark.parametrize("use_fire_reset", [False, True])
    def test_determinism(
        self,
        env_id,
        num_envs: int,
        seeds: np.ndarray,
        noop_max: int,
        repeat_action_probability: float,
        use_fire_reset: bool,
        rollout_length: int = 100,
    ):
        envs_1 = gym.make_vec(
            env_id,
            num_envs,
            noop_max=noop_max,
            repeat_action_probability=repeat_action_probability,
            use_fire_reset=use_fire_reset,
        )
        envs_2 = gym.make_vec(
            env_id,
            num_envs,
            noop_max=noop_max,
            repeat_action_probability=repeat_action_probability,
            use_fire_reset=use_fire_reset,
        )

        obs_1, info_1 = envs_1.reset(seed=seeds)
        obs_2, info_2 = envs_2.reset(seed=seeds)

        assert data_equivalence(obs_1, obs_2)
        assert data_equivalence(info_1, info_2)

        for t in range(rollout_length):
            actions = envs_1.action_space.sample()

            obs_1, rewards_1, terminations_1, truncations_1, info_1 = envs_1.step(
                actions
            )
            obs_2, rewards_2, terminations_2, truncations_2, info_2 = envs_2.step(
                actions
            )

            assert data_equivalence(obs_1, obs_2)
            assert data_equivalence(rewards_1, rewards_2)
            assert data_equivalence(terminations_1, terminations_2)
            assert data_equivalence(truncations_1, truncations_2)
            assert data_equivalence(info_1, info_2)

        envs_1.close()
        envs_2.close()

    def test_batch_size_async(
        self,
        env_id,
        batch_size=4,
        num_envs=8,
        rollout_length=1000,
        reset_seed=123,
        action_seed=123,
    ):
        """Tests asynchronous feature of the vector environment.

        Using a batch_size < num_envs then the first N sub-environments results are returned.
        We use the synchronous (all sub-environments) as the baseline and compare a sub-environment's results
           to for the synchronous's result for the same action.
        """
        sync_envs = gym.make_vec(env_id, num_envs)
        async_envs = gym.make_vec(env_id, num_envs, batch_size=batch_size)
        assert sync_envs.num_envs == async_envs.num_envs

        assert sync_envs.single_action_space == async_envs.single_action_space
        assert sync_envs.single_observation_space == async_envs.single_observation_space
        assert sync_envs.action_space != async_envs.action_space
        assert sync_envs.observation_space != async_envs.observation_space

        sync_envs.action_space.seed(action_seed)
        actions = [sync_envs.action_space.sample() for _ in range(rollout_length)]
        async_env_timestep = np.zeros(num_envs, dtype=np.int32)

        sync_obs, sync_info = sync_envs.reset(seed=reset_seed)
        sync_env_ids = sync_info.pop("env_id")
        assert np.all(sync_env_ids == np.arange(num_envs)), f"{sync_env_ids=}"
        async_obs, async_info = async_envs.reset(seed=reset_seed)
        async_env_ids = async_info.pop("env_id")
        assert async_env_ids.shape == (batch_size,), f"{async_env_ids=}"

        sync_observations = [sync_obs]
        sync_rewards = [np.zeros(num_envs, dtype=np.int32)]
        sync_terminations = [np.zeros(num_envs, dtype=bool)]
        sync_truncations = [np.zeros(num_envs, dtype=bool)]
        sync_infos = [sync_info]

        for async_i, env_id in enumerate(async_env_ids):
            async_t = async_env_timestep[env_id]
            assert data_equivalence(
                sync_observations[async_t][env_id], async_obs[async_i]
            )
            assert all(
                data_equivalence(
                    sync_infos[async_t][key][env_id], async_info[key][async_i]
                )
                for key in sync_info
            )
        async_env_timestep[async_env_ids] += 1

        for t in range(rollout_length):
            obs, rewards, terminations, truncations, info = sync_envs.step(actions[t])
            sync_observations.append(obs)
            sync_rewards.append(rewards)
            sync_terminations.append(terminations)
            sync_truncations.append(truncations)
            sync_env_ids = info.pop("env_id")
            assert np.all(sync_env_ids == np.arange(num_envs)), f"{sync_env_ids=}"
            sync_infos.append(info)

            async_actions = np.array(
                [
                    actions[async_env_timestep[env_id] - 1][env_id]
                    for env_id in async_env_ids
                ]
            )
            (
                async_obs,
                async_rewards,
                async_terminations,
                async_truncations,
                async_info,
            ) = async_envs.step(async_actions)
            async_env_ids = async_info.pop("env_id")
            assert async_env_ids.shape == (batch_size,), f"{async_env_ids=}"

            for async_i, env_id in enumerate(async_env_ids):
                async_t = async_env_timestep[env_id]

                assert data_equivalence(
                    sync_observations[async_t][env_id], async_obs[async_i]
                )
                if not data_equivalence(
                    sync_rewards[async_t][env_id], async_rewards[async_i]
                ):
                    print(
                        f"{sync_rewards[async_t][env_id]=}, {async_rewards[async_i]=}, {async_t=}, {env_id=}, {async_i=}"
                    )
                    print(
                        f"{type(sync_rewards[async_t][env_id])=}, {type(async_rewards[async_i])=}"
                    )
                    print(
                        f"{data_equivalence(sync_rewards[async_t][env_id], async_rewards[async_i])=}"
                    )
                    print(
                        f"{sync_rewards[async_t][env_id].dtype=}, {async_rewards[async_i].dtype=}"
                    )
                    print(
                        f"{sync_rewards[async_t][env_id].shape=}, {async_rewards[async_i].shape=}"
                    )
                    print(f"{sync_rewards[async_t][env_id] == async_rewards[async_i]=}")
                    print(
                        f"{np.all(sync_rewards[async_t][env_id] == async_rewards[async_i])=}"
                    )
                assert data_equivalence(
                    sync_terminations[async_t][env_id], async_terminations[async_i]
                )
                assert data_equivalence(
                    sync_truncations[async_t][env_id], async_truncations[async_i]
                )
                assert all(
                    data_equivalence(
                        sync_infos[async_t][key][env_id], async_info[key][async_i]
                    )
                    for key in sync_info
                )
            async_env_timestep[async_env_ids] += 1

        sync_envs.close()
        async_envs.close()

    def test_episodic_life_equivalence(self, env_id, num_envs=8):
        gym_envs = gym.vector.SyncVectorEnv(
            [
                lambda: gym.wrappers.FrameStackObservation(
                    gym.wrappers.AtariPreprocessing(
                        gym.make(env_id, **self.disable_env_args),
                        terminal_on_life_loss=True,
                        **self.disable_preprocessing_args,
                    ),
                    stack_size=4,
                    padding_type="zero",
                )
                for _ in range(num_envs)
            ],
        )
        ale_envs = gym.make_vec(
            env_id, num_envs, episodic_life=True, **self.disable_vector_args
        )

        assert_gym_ale_rollout_equivalence(gym_envs, ale_envs, episodic_life=True)

    def test_episodic_life_and_life_loss_info(
        self, env_id, num_envs=8, rollout_length=1000, reset_seed=123, action_seed=123
    ):
        standard_envs = gym.make_vec(env_id, num_envs)
        episodic_life_envs = gym.make_vec(env_id, num_envs, episodic_life=True)
        life_loss_envs = gym.make_vec(env_id, num_envs, life_loss_info=True)

        standard_envs.action_space.seed(action_seed)
        standard_obs, standard_info = standard_envs.reset(seed=reset_seed)
        episodic_life_obs, episodic_life_info = episodic_life_envs.reset(
            seed=reset_seed
        )
        life_loss_obs, life_loss_info = life_loss_envs.reset(seed=reset_seed)

        assert data_equivalence(standard_obs, episodic_life_obs)
        assert data_equivalence(standard_obs, life_loss_obs)
        assert data_equivalence(standard_info, episodic_life_info)
        assert data_equivalence(standard_info, life_loss_info)

        previous_lives = standard_info["lives"]
        if np.all(previous_lives == 0):
            actions = standard_envs.action_space.sample()
            _, _, standard_terminations, _, _ = standard_envs.step(actions)
            _, _, life_loss_terminations, _, _ = life_loss_envs.step(actions)
            _, _, episodic_life_terminations, _, _ = episodic_life_envs.step(actions)

            assert not np.any(standard_terminations)
            assert not np.any(life_loss_terminations)
            assert not np.any(episodic_life_terminations)

            return  # no more testing can be done for this environment

        rollout_life_lost = False
        for t in range(rollout_length):
            actions = standard_envs.action_space.sample()

            (
                standard_obs,
                standard_rewards,
                standard_terminations,
                standard_truncations,
                standard_info,
            ) = standard_envs.step(actions)
            (
                life_loss_obs,
                life_loss_rewards,
                life_loss_terminations,
                life_loss_truncations,
                life_loss_info,
            ) = life_loss_envs.step(actions)

            lives = standard_info["lives"]
            action_life_lost = np.logical_and(previous_lives > lives, lives > 0)

            assert obs_equivalence(standard_obs, life_loss_obs, t=t, life_loss=True)
            assert data_equivalence(standard_rewards, life_loss_rewards)
            assert np.all(
                np.logical_or(standard_terminations, action_life_lost)
                == life_loss_terminations
            ), f"{standard_terminations=}, {action_life_lost=}, {life_loss_terminations=}, {t=}, {standard_info=}, {life_loss_info=}"
            assert data_equivalence(standard_truncations, life_loss_truncations)
            assert data_equivalence(standard_info, life_loss_info)

            if not rollout_life_lost:
                rollout_life_lost = life_loss_terminations.any()

                (
                    episodic_life_obs,
                    episodic_life_rewards,
                    episodic_life_terminations,
                    episodic_life_truncations,
                    episodic_life_info,
                ) = episodic_life_envs.step(actions)

                # Due to ending the frame skip early if a life is loss (following AtariPreprocessing)
                #    then the observations, rewards and info frame number might not be equivalent
                # assert obs_equivalence(
                #     standard_obs, episodic_life_obs, i=t, episodic_life=True
                # )
                # assert data_equivalence(standard_rewards, episodic_life_rewards)
                assert np.all(
                    np.logical_or(standard_terminations, action_life_lost)
                    == episodic_life_terminations
                ), f"{standard_terminations=}, {action_life_lost=}, {episodic_life_terminations=}, {t=}"
                assert data_equivalence(standard_truncations, episodic_life_truncations)
                # assert data_equivalence(standard_info, episodic_life_info)

            previous_lives = standard_info["lives"]

        if not rollout_life_lost:
            gym.logger.warn(
                f"No life lost in rollout for {env_id} in test_episodic_life_and_life_loss_info"
            )
        # assert rollout_life_lost

        standard_envs.close()
        episodic_life_envs.close()
        life_loss_envs.close()

    def test_same_step_autoreset_mode(
        self, env_id, num_envs=4, reset_seed=123, action_seed=123, rollout_length=100
    ):
        """Test if both environments produce similar results over a short rollout."""
        gym_envs = gym.vector.SyncVectorEnv(
            [
                lambda: gym.wrappers.FrameStackObservation(
                    gym.wrappers.AtariPreprocessing(
                        gym.make(
                            env_id,
                            **self.disable_env_args,
                        ),
                        terminal_on_life_loss=True,  # to ensure some terminations
                        **self.disable_preprocessing_args,
                    ),
                    stack_size=4,
                    padding_type="zero",
                )
                for _ in range(num_envs)
            ],
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        )
        ale_envs = gym.make_vec(
            env_id,
            num_envs,
            episodic_life=True,
            autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
            **self.disable_vector_args,
        )
        assert (
            gym_envs.metadata["autoreset_mode"] == ale_envs.metadata["autoreset_mode"]
        ), f"{gym_envs.metadata=}, {ale_envs.metadata=}"

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
        has_autoreset = False
        for t in range(rollout_length):
            actions = ale_envs.action_space.sample()

            gym_obs, gym_rewards, gym_terminations, gym_truncations, gym_info = (
                gym_envs.step(actions)
            )
            ale_obs, ale_rewards, ale_terminations, ale_truncations, ale_info = (
                ale_envs.step(actions)
            )

            assert obs_equivalence(gym_obs, ale_obs, t, autoreset_mode="SAME-STEP"), t
            assert data_equivalence(gym_rewards.astype(np.int32), ale_rewards), t
            assert data_equivalence(gym_terminations, ale_terminations), t
            assert data_equivalence(gym_truncations, ale_truncations), t

            env_ids = ale_info.pop("env_id")
            assert np.all(env_ids == np.arange(gym_envs.num_envs)), t

            episode_over = np.logical_or(gym_terminations, gym_truncations)
            if np.any(episode_over):
                has_autoreset = True

                gym_final_obs = np.array(
                    [
                        final_obs if ep_over else obs
                        for final_obs, obs, ep_over in zip(
                            gym_info.pop("final_obs"), gym_obs, episode_over
                        )
                    ]
                )
                gym_info.pop("final_info")  # ALEV doesn't return final info
                gym_info = {
                    key: value.astype(np.int32)
                    for key, value in gym_info.items()
                    if not key.startswith("_")
                }

                ale_final_obs = ale_info.pop("final_obs")
                assert data_equivalence(
                    gym_info, ale_info
                ), f"{gym_info=}, {ale_info=}, {t=}"

                assert obs_equivalence(
                    gym_final_obs, ale_final_obs, t, autoreset_mode="SAME-STEP"
                ), t
            else:
                gym_info = {
                    key: value.astype(np.int32)
                    for key, value in gym_info.items()
                    if not key.startswith("_") and key != "seeds"
                }

                assert data_equivalence(gym_info, ale_info), t

        assert has_autoreset

        gym_envs.close()
        ale_envs.close()
