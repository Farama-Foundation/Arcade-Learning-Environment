from typing import Any

import ale_py
import gymnasium.vector.utils
import numpy as np
import roms
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete
from gymnasium.vector import VectorEnv


class AtariVectorEnv(VectorEnv):

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
        zero_discount_on_life_loss: bool = False,
        reward_clip: int = True,
        use_fire_reset: bool = True,
    ):

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
            zero_discount_on_life_loss=zero_discount_on_life_loss,
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
        if options is None or "reset_mask" not in options:
            reset_mask = np.arange(self.num_envs)
        else:
            reset_mask = options["reset_mask"]
        return self.ale.reset(reset_mask)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        self.ale.send(actions)
        return self.ale.recv()

    def send(self, actions: np.ndarray):
        # The actions will be the size of the batch_size
        self.ale.send(actions)

    def recv(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # The data will be of the batch_size, see `info["env_id"]` for the set of environments used.
        return self.ale.recv()

    def close(self, **kwargs: Any):
        del self.ale
