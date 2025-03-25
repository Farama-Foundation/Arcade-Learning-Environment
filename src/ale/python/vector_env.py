"""Vector environment for ALE."""

from typing import Any

import ale_py
import gymnasium.vector.utils
import numpy as np
import roms
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import VectorEnv


class AtariVectorEnv(VectorEnv):
    """Vector environment implementation for ALE."""

    def __init__(
        self,
        game: str,
        num_envs: int,
        batch_size: int = 0,
        num_threads: int = 0,
        thread_affinity_offset: int = -1,
        max_episode_steps: int = 27000,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        # Preprocessing values
        obs_height: int = 84,
        obs_width: int = 84,
        stack_num: int = 4,
        grayscale: bool = True,
        frame_skip: int = 4,
        noop_max: int = 30,
        episodic_life: bool = False,
        life_loss_information: bool = False,
        reward_clip: int = True,
        use_fire_reset: bool = True,
    ):
        """Constructor for vector environment.

        Args:
            game: ROM name
            num_envs: Number of environments
            batch_size: If to provide a batch of environments (in async mode)
            num_threads: The number of threads to use for parallel environments
            thread_affinity_offset: TODO
            max_episode_steps: Maximum number of steps per episode
            repeat_action_probability: Repeat action probability for the sub-environments
            full_action_space: If the environment should use the full action space
            obs_height: The observation height
            obs_width: The observation width
            stack_num: The stack size
            grayscale: If to use grayscale observations
            frame_skip: The number of frame skips to use for each action
            noop_max: If to use noop-max for the episode resets
            episodic_life: If to terminate episodes on life losses
            life_loss_information: If to provide a termination signal on life loss
            reward_clip: If to clip rewards between -1 and 1
            use_fire_reset: If to take fire action on reset if available
        """
        self.ale = ale_py.vector.ALEVectorInterface(
            rom_path=roms.get_rom_path(game),
            num_envs=num_envs,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
            max_episode_steps=max_episode_steps,
            repeat_action_probability=repeat_action_probability,
            full_action_space=full_action_space,
            obs_height=obs_height,
            obs_width=obs_width,
            stack_num=stack_num,
            grayscale=grayscale,
            frame_skip=frame_skip,
            noop_max=noop_max,
            episodic_life=episodic_life,
            zero_discount_on_life_loss=life_loss_information,
            reward_clip=reward_clip,
            use_fire_reset=use_fire_reset,
        )

        obs_shape = (stack_num, obs_height, obs_width)
        if not grayscale:
            obs_shape += (3,)
        self.single_observation_space = Box(
            shape=obs_shape, low=0, high=255, dtype=np.uint8
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
        if options is None or "reset_mask" not in options:
            reset_mask = np.arange(self.num_envs)
        else:
            reset_mask = options["reset_mask"]
        return self.ale.reset(reset_mask)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Steps through the sub-environments for which the actions are taken, return arrays for the next observations, rewards, termination, truncation and info."""
        self.ale.send(actions)
        return self.ale.recv()

    def send(self, actions: np.ndarray):
        """Send the actions to the sub-environments."""
        # The actions will be the size of the batch_size
        self.ale.send(actions)

    def recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Receive the next observations, rewards, terminations, truncations and info from the sub-environments."""
        # The data will be of the batch_size, see `info["env_id"]` for the set of environments used.
        return self.ale.recv()

    def close(self, **kwargs: Any):
        """Destroys the vectoriser."""
        del self.ale
