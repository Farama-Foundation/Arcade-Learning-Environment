"""Vector environment for ALE."""

from __future__ import annotations

from typing import Any

import ale_py
import gymnasium.vector.utils
import numpy as np
from ale_py import roms
from ale_py.env import AtariEnv
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import VectorEnv


class AtariVectorEnv(VectorEnv):
    """Vector environment implementation for ALE."""

    def __init__(
        self,
        game: str,
        num_envs: int,
        *,
        batch_size: int = 0,
        num_threads: int = 0,
        thread_affinity_offset: int = -1,
        max_num_frames_per_episode: int = 108000,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        continuous: bool = False,
        continuous_action_threshold: float = 0.5,
        # Preprocessing values
        img_height: int = 84,
        img_width: int = 84,
        grayscale: bool = True,
        stack_num: int = 4,
        frameskip: int = 4,
        maxpool: bool = True,
        noop_max: int = 30,
        episodic_life: bool = False,
        life_loss_info: bool = False,
        reward_clipping: bool = True,
        use_fire_reset: bool = True,
    ):
        """Constructor for vector environment.

        Args:
            game: ROM name
            num_envs: Number of environments
            batch_size: If to provide a batch of environments (in async mode)
            num_threads: The number of threads to use for parallel environments
            thread_affinity_offset: The CPU core offset for thread affinity (-1 means no affinity, default: -1)
            max_num_frames_per_episode: Maximum number of steps per episode
            repeat_action_probability: Repeat action probability for the sub-environments
            full_action_space: If the environment should use the full action space
            continuous: If to use the continuous action space
            continuous_action_threshold: The continuous action threshold
            img_height: The frame height
            img_width: The frame width
            grayscale: Whether to use grayscale observations
            stack_num: The frame stack size
            frameskip: The number of frame skips to use for each action
            maxpool: If maxpool over subsequent frames
            noop_max: If to use noop-max for the episode resets
            episodic_life: If to terminate episodes on life losses
            life_loss_info: If to provide a termination signal on life loss
            reward_clipping: If to clip rewards between -1 and 1
            use_fire_reset: If to take fire action on reset if available
        """
        self.ale = ale_py.ALEVectorInterface(
            rom_path=roms.get_rom_path(game),
            num_envs=num_envs,
            frame_skip=frameskip,
            stack_num=stack_num,
            img_height=img_height,
            img_width=img_width,
            grayscale=grayscale,
            maxpool=maxpool,
            noop_max=noop_max,
            use_fire_reset=use_fire_reset,
            episodic_life=episodic_life,
            life_loss_info=life_loss_info,
            reward_clipping=reward_clipping,
            max_episode_steps=max_num_frames_per_episode,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
        )

        self.continuous = continuous
        self.continuous_action_threshold = continuous_action_threshold
        self.grayscale = grayscale
        self.map_action_idx = np.zeros((3, 3, 2), dtype=np.int32)
        for h in (-1, 0, 1):
            for v in (-1, 0, 1):
                for f in (False, True):
                    self.map_action_idx[h, v, f] = AtariEnv.map_action_idx(
                        h, v, f
                    ).value

        # Set up the observation space based on grayscale or RGB format
        obs_shape = (stack_num, img_height, img_width)
        if not grayscale:
            obs_shape += (3,)
        self.single_observation_space = Box(
            shape=obs_shape, low=0, high=255, dtype=np.uint8
        )

        if self.continuous:
            # Actions are radius, theta, and fire, where first two are the parameters of polar coordinates.
            self.single_action_space = Box(
                low=np.array([0.0, -np.pi, 0.0]).astype(np.float32),
                high=np.array([1.0, np.pi, 1.0]).astype(np.float32),
                dtype=np.float32,
                shape=(3,),
            )
        else:
            self.single_action_space = Discrete(len(self.ale.get_action_set()))

        self.batch_size = num_envs if batch_size == 0 else batch_size
        self.num_envs = num_envs
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, self.batch_size
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space, self.batch_size
        )

        self.is_xla_registered = False

    def reset(
        self,
        *,
        seed: int | np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the sub-environments.

        Args:
            seed: Current unimplemented
            options: Supports `reset_mask` that indicates what sub-environments should be reset

        Returns:
            Tuple of observations for the sub-environments and info on them.
        """
        if options is None or "reset_mask" not in options:
            reset_indices = np.arange(self.num_envs)
        else:
            reset_mask = options["reset_mask"]
            assert isinstance(reset_mask, np.ndarray) and reset_mask.dtype == np.bool_
            reset_indices, _ = np.where(reset_mask)

        if seed is None:
            reset_seeds = np.full(len(reset_indices), -1)
        elif isinstance(seed, int):
            reset_seeds = np.arange(seed, seed + len(reset_indices))
        elif isinstance(seed, np.ndarray):
            reset_seeds = seed
        else:
            raise TypeError("Unsupported seed type")

        return self.ale.reset(reset_indices, reset_seeds)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Steps through the sub-environments for which the actions are taken, return arrays for the next observations, rewards, termination, truncation and info."""
        self.send(actions)
        return self.ale.recv()

    def send(self, actions: np.ndarray):
        """Send the actions to the sub-environments."""
        if self.continuous:
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.float32
            assert actions.shape == (self.batch_size, 3)

            x = actions[:, 0] * np.cos(actions[:, 1])
            y = actions[:, 0] * np.sin(actions[:, 1])

            horizontal = -(x < self.continuous_action_threshold) + (
                x > self.continuous_action_threshold
            )
            vertical = -(y < self.continuous_action_threshold) + (
                y > self.continuous_action_threshold
            )
            fire = actions[:, 2] > self.continuous_action_threshold

            action_ids = self.map_action_idx[np.array([horizontal, vertical, fire])]
            paddle_strength = actions[:, 0]
            self.ale.send(action_ids, paddle_strength)
        else:
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.int64 or actions.dtype == np.int32
            assert actions.shape == (self.batch_size,)

            paddle_strength = np.ones(self.batch_size)
            self.ale.send(actions, paddle_strength)

    def recv(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Receive the next observations, rewards, terminations, truncations and info from the sub-environments."""
        # The data will be of the batch_size, see `info["env_id"]` for the set of environments used.
        return self.ale.recv()

    def xla(self):
        """Return XLA-compatible functions for JAX integration."""
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                'ALE is missing jax, necessary for using the vector XLA support, use `pip install "ale_py[xla]"` to import'
            ) from e

        if not self.is_xla_registered:
            jax.ffi.register_ffi_target(
                "atari_vector_xla_reset",
                ale_py._ale_py.VectorXLAReset(),
            )
            jax.ffi.register_ffi_target(
                "atari_vector_xla_step",
                ale_py._ale_py.VectorXLAStep(),
            )
            self.is_xla_registered = True

        def xla_reset(
            handle: np.ndarray,
            seed: np.ndarray | None = None,
            reset_mask: np.ndarray | None = None,
        ) -> tuple[np.ndarray, tuple[np.ndarray, dict[str, Any]]]:
            xla_call = jax.ffi.ffi_call(
                target_name="atari_vector_xla_reset",
                result_shape_dtypes=(
                    jax.ShapeDtypeStruct((8,), jnp.uint8),  # handle
                    jax.ShapeDtypeStruct(
                        self.observation_space.shape, jnp.uint8
                    ),  # observations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # env_ids
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # lives
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # frame numbers
                    jax.ShapeDtypeStruct(
                        (self.num_envs,), jnp.int32
                    ),  # episode frame number
                ),
                vmap_method="broadcast_all",
            )

            if reset_mask is not None:
                assert (
                    isinstance(reset_mask, np.ndarray) and reset_mask.dtype == np.bool_
                )
                reset_indices, _ = np.where(reset_mask)
            else:
                reset_indices = np.arange(self.num_envs)

            if seed is None:
                reset_seeds = np.full(len(reset_indices), -1)
            elif isinstance(seed, int):
                reset_seeds = np.arange(seed, seed + len(reset_indices))
            elif isinstance(seed, np.ndarray):
                reset_seeds = seed
            else:
                raise TypeError("Unsupported seed type")

            new_handle, obs, env_ids, lives, frame_numbers, episode_frame_numbers = (
                xla_call(handle, reset_indices, reset_seeds)
            )

            info = {
                "env_id": np.asarray(env_ids),
                "lives": np.asarray(lives),
                "frame_number": np.asarray(frame_numbers),
                "episode_frame_number": np.asarray(episode_frame_numbers),
            }
            return new_handle, (np.asarray(obs), info)

        def xla_step(handle, actions):
            if self.continuous:
                assert isinstance(actions, np.ndarray)
                assert actions.dtype == np.float32
                assert actions.shape == (self.batch_size, 2)

                x = actions[:, 0] * np.cos(actions[:, 1])
                y = actions[:, 0] * np.sin(actions[:, 1])

                horizontal = -(x < self.continuous_action_threshold) + (
                    x > self.continuous_action_threshold
                )
                vertical = -(y < self.continuous_action_threshold) + (
                    y > self.continuous_action_threshold
                )
                fire = actions[:, 1] > self.continuous_action_threshold

                action_ids = self.map_action_idx[np.array([horizontal, vertical, fire])]
                paddle_strength = actions[:, 1]
            else:
                assert isinstance(actions, np.ndarray)
                assert actions.dtype == np.int64 or actions.dtype == np.int32
                assert actions.shape == (self.batch_size,)

                action_ids = actions
                paddle_strength = np.ones(self.batch_size)

            xla_call = jax.ffi.ffi_call(
                target_name="atari_vector_xla_step",
                result_shape_dtypes=(
                    jax.ShapeDtypeStruct((8,), jnp.uint8),  # handle
                    jax.ShapeDtypeStruct(
                        self.observation_space.shape, jnp.uint8
                    ),  # observations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # rewards
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),  # terminations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),  # truncations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # env_ids
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # lives
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # frame numbers
                    jax.ShapeDtypeStruct(
                        (self.num_envs,), jnp.int32
                    ),  # episode frame number
                ),
                vmap_method="broadcast_all",
            )

            (
                new_handle,
                obs,
                rewards,
                terminations,
                truncations,
                env_ids,
                lives,
                frame_numbers,
                episode_frame_numbers,
            ) = xla_call(handle, action_ids, paddle_strength)

            info = {
                "env_id": np.asarray(env_ids),
                "lives": np.asarray(lives),
                "frame_number": np.asarray(frame_numbers),
                "episode_frame_number": np.asarray(episode_frame_numbers),
            }
            return new_handle, (
                np.asarray(obs),
                np.asarray(rewards),
                np.asarray(terminations),
                np.asarray(truncations),
                info,
            )

        # Get the vectorizer handle and make sure it's properly formatted
        ale_handle = np.frombuffer(self.ale.handle(), dtype=np.uint8)
        return ale_handle, xla_reset, xla_step
