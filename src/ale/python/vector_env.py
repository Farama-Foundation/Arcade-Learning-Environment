from typing import Any

import numpy as np
from ale_py import roms
from ale_py._ale_py.vector import ALEVectorInterface
from gymnasium.core import ActType, ObsType
from gymnasium.spaces import MultiDiscrete, Box
from gymnasium.vector import VectorEnv
from gymnasium.vector.vector_env import ArrayType


class AtariVectorEnv(VectorEnv):

    def __init__(self, game: str, num_envs: int):

        self.ale = ALEVectorInterface(num_envs, roms.get_rom_path(game))
        self.observation_space = Box(
            shape=(num_envs, 84, 84, 1), low=0, high=255, dtype=np.uint8
        )
        self.action_space = MultiDiscrete([len(self.ale.get_action_set()) for _ in range(num_envs)])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        return self.ale.reset()

    def step(self, actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        return 0

    def close(self, **kwargs: Any):
        del self.ale
