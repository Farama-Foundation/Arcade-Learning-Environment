"""Vector environment for ALE."""

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
        max_episode_steps: int = 27000,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        # Preprocessing values
        img_height: int = 84,
        img_width: int = 84,
        stack_num: int = 4,
        frame_skip: int = 4,
        noop_max: int = 30,
        episodic_life: bool = False,
        life_loss_info: bool = False,
        reward_clipping: bool = True,
        use_fire_reset: bool = True,
        continuous: bool = False,
        continuous_action_threshold: float = 0.5,
    ):
        """Constructor for vector environment.

        Args:
            game: ROM name
            num_envs: Number of environments
            batch_size: If to provide a batch of environments (in async mode)
            num_threads: The number of threads to use for parallel environments
            thread_affinity_offset: The CPU core offset for thread affinity (-1 means no affinity, default: -1)
            max_episode_steps: Maximum number of steps per episode
            repeat_action_probability: Repeat action probability for the sub-environments
            full_action_space: If the environment should use the full action space
            img_height: The frame height
            img_width: The ƒrame width
            stack_num: The frame stack size
            frame_skip: The number of frame skips to use for each action
            noop_max: If to use noop-max for the episode resets
            episodic_life: If to terminate episodes on life losses
            life_loss_info: If to provide a termination signal on life loss
            reward_clipping: If to clip rewards between -1 and 1
            use_fire_reset: If to take fire action on reset if available
            continuous: If to use the continuous action space
            continuous_action_threshold: The continuous action threshold
        """
        self.ale = ale_py.ALEVectorInterface(
            rom_path=roms.get_rom_path(game),
            num_envs=num_envs,
            frame_skip=frame_skip,
            stack_num=stack_num,
            img_height=img_height,
            img_width=img_width,
            noop_max=noop_max,
            use_fire_reset=use_fire_reset,
            episodic_life=episodic_life,
            life_loss_info=life_loss_info,
            reward_clipping=reward_clipping,
            max_episode_steps=max_episode_steps,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
        )

        self.single_observation_space = Box(
            shape=(stack_num, img_height, img_width), low=0, high=255, dtype=np.uint8
        )
        self.single_action_space = Discrete(len(self.ale.get_action_set()))

        self.batch_size = num_envs if batch_size == 0 else batch_size
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, batch_size
        )
        self.action_space = gymnasium.vector.utils.batch_space(
            self.single_action_space, batch_size
        )
        self.num_envs = num_envs
        self.batch_size = batch_size if batch_size > 0 else num_envs

        self.continuous = continuous
        self.continuous_action_threshold = continuous_action_threshold
        self.map_action_idx = np.zeros((3, 3, 2), dtype=np.int32)
        for h in (-1, 0, 1):
            for v in (-1, 0, 1):
                for f in (False, True):
                    self.map_action_idx[h, v, f] = AtariEnv.map_action_idx(
                        h, v, f
                    ).value

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the sub-environments.

        Args:
            seed: Current unimplemented
            options: Supports `reset_mask` that indicates what sub-environments should be reset

        Returns:
            Tuple of observations for the sub-environments and info on them.
        """
        # todo deal with seed

        if options is None or "reset_mask" not in options:
            reset_indices = np.arange(self.num_envs)
        else:
            reset_mask = options["reset_mask"]
            assert isinstance(reset_mask, np.ndarray) and reset_mask.dtype == np.bool
            reset_indices, _ = np.where(reset_mask)

        return self.ale.reset(reset_indices)

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
            assert actions.shape == (self.batch_size, 2)

            x = actions[0, :] * np.cos(actions[1, :])
            y = actions[0, :] * np.sin(actions[1, :])

            horizontal = -(x < self.continuous_action_threshold) + (
                x > self.continuous_action_threshold
            )
            vertical = -(y < self.continuous_action_threshold) + (
                y > self.continuous_action_threshold
            )
            fire = actions[1, :] > self.continuous_action_threshold

            action_ids = self.map_action_idx[horizontal, vertical, fire]
            paddle_strength = actions[1, :]
            self.ale.send(action_ids, paddle_strength)
        else:
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.int32
            assert actions.shape == (self.batch_size,)

            paddle_strength = np.ones(self.batch_size)
            self.ale.send(actions, paddle_strength)

    def recv(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Receive the next observations, rewards, terminations, truncations and info from the sub-environments."""
        # The data will be of the batch_size, see `info["env_id"]` for the set of environments used.
        return self.ale.recv()
