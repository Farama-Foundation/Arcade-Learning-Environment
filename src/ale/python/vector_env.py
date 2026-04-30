"""Vector environment for ALE."""

from __future__ import annotations

from typing import Any

import ale_py
import gymnasium.vector.utils
import numpy as np
from ale_py import roms
from ale_py.env import AtariEnv
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple
from gymnasium.vector import AutoresetMode, VectorEnv


class AtariVectorEnv(VectorEnv):
    """Vector environment implementation for ALE."""

    def __init__(
        self,
        game: str | list[str],
        num_envs: int | None = None,
        *,
        batch_size: int = 0,
        num_threads: int = 0,
        thread_affinity_offset: int = -1,
        max_num_frames_per_episode: int = 108000,
        repeat_action_probability: float = 0.0,
        full_action_space: bool = False,
        continuous: bool = False,
        continuous_action_threshold: float = 0.5,
        autoreset_mode: AutoresetMode | str = AutoresetMode.NEXT_STEP,
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
            game: ROM name (e.g. ``"breakout"``) or list of ROM names for multi-ROM mode; num_envs is inferred from list length
            num_envs: Number of environments (required when game is a str)
            batch_size: If to provide a batch of environments (in async mode)
            num_threads: The number of threads to use for parallel environments
            thread_affinity_offset: The CPU core offset for thread affinity (-1 means no affinity, default: -1)
            max_num_frames_per_episode: Maximum number of steps per episode
            repeat_action_probability: Repeat action probability for the sub-environments
            full_action_space: If the environment should use the full action space
            continuous: If to use the continuous action space
            continuous_action_threshold: The continuous action threshold
            autoreset_mode: What mode to autoreset the sub-environments
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
        _autoreset_str = (
            autoreset_mode.value
            if isinstance(autoreset_mode, AutoresetMode)
            else autoreset_mode
        )
        _common = dict(
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
            full_action_space=full_action_space or continuous,
            batch_size=batch_size,
            num_threads=num_threads,
            thread_affinity_offset=thread_affinity_offset,
            autoreset_mode=_autoreset_str,
        )

        if isinstance(game, list):
            rom_paths = []
            for g in game:
                rp = roms.get_rom_path(g)
                assert rp is not None, (
                    f"{g} is not a ROM name, it should be snake_case not camel-case, "
                    f'i.e., "ms_pacman" not "MsPacman"'
                )
                rom_paths.append(rp)
            if num_envs is not None:
                assert num_envs == len(
                    game
                ), f"num_envs={num_envs} does not match len(game)={len(game)}"
            num_envs = len(game)
            self.ale = ale_py.ALEVectorInterface(rom_paths=rom_paths, **_common)
        else:
            assert num_envs is not None, "num_envs is required when game is a str"
            rom_path = roms.get_rom_path(game)
            assert rom_path is not None, (
                f"{game} is not a ROM name, it should be snake_case not camel-case, "
                f'i.e., "ms_pacman" not "MsPacman"'
            )
            self.ale = ale_py.ALEVectorInterface(
                rom_path=rom_path, num_envs=num_envs, **_common
            )

        self.num_actions: list[int] = self.ale.num_actions()
        self.action_set: list[list[int]] = [
            [a.value for a in s] for s in self.ale.get_action_set()
        ]

        self.continuous = continuous
        self.continuous_action_threshold = continuous_action_threshold
        self.grayscale = grayscale
        self.map_action_idx = np.zeros((3, 3, 2), dtype=np.int32)
        for h in (-1, 0, 1):
            for v in (-1, 0, 1):
                for f in (0, 1):
                    action = AtariEnv.map_action_idx(h, v, bool(f)).value
                    self.map_action_idx[h + 1, v + 1, f] = action

        # Set up the observation space based on grayscale or RGB format
        obs_shape = (stack_num, img_height, img_width)
        if not grayscale:
            obs_shape += (3,)
        self.single_observation_space = Box(
            shape=obs_shape, low=0, high=255, dtype=np.uint8
        )

        self.batch_size = num_envs if batch_size == 0 else batch_size
        self.num_envs = num_envs
        self.metadata["autoreset_mode"] = (
            autoreset_mode
            if isinstance(autoreset_mode, AutoresetMode)
            else AutoresetMode(autoreset_mode)
        )
        self.observation_space = gymnasium.vector.utils.batch_space(
            self.single_observation_space, self.batch_size
        )

        if self.continuous:
            self.single_action_space = Box(
                low=np.array([0.0, -np.pi, 0.0]).astype(np.float32),
                high=np.array([1.0, np.pi, 1.0]).astype(np.float32),
                dtype=np.float32,
                shape=(3,),
            )
            self.action_space = gymnasium.vector.utils.batch_space(
                self.single_action_space, self.batch_size
            )
        elif len(set(self.num_actions)) == 1:
            self.single_action_space = Discrete(self.num_actions[0])
            self.action_space = MultiDiscrete(
                np.full(self.batch_size, self.num_actions[0], dtype=np.int64)
            )
        else:
            self.single_action_space = Discrete(max(self.num_actions))
            self.action_space = Tuple(
                tuple(Discrete(n) for n in self.num_actions[: self.batch_size])
            )

    def reset(
        self,
        *,
        seed: int | np.ndarray | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, np.ndarray]]:
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
            (reset_indices,) = np.where(reset_mask)

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
        self,
        actions: int | np.integer | np.ndarray | list[Any],
        gamma: float | list[float] = 1.0,
        paddle_strength: float | list[float] = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Send actions and receive a new observation in an atomic call.

        Pass a flat array for single-step mode or a list of arrays for multi-step
        mode (one array of action sequences per rom). info includes
        ``steps_taken`` for multi-step mode.
        """
        self.send(actions, gamma=gamma, paddle_strength=paddle_strength)
        return self.recv()

    def send(
        self,
        actions: int | np.integer | np.ndarray | list[Any],
        gamma: float | list[float] = 1.0,
        paddle_strength: float | list[float] = 1.0,
    ):
        """Dispatch actions to roms without waiting for results.

        For multi-step mode, pass a list of arrays where each array is the action
        sequence for one rom. ``gamma`` and ``paddle_strength`` can be
        scalars or per-rom lists.
        """
        if self.continuous:
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.float32
            assert actions.shape == (self.batch_size, 3)

            x = actions[:, 0] * np.cos(actions[:, 1])
            y = actions[:, 0] * np.sin(actions[:, 1])

            horizontal = (
                -(x < -self.continuous_action_threshold).astype(np.int32)
                + (x > self.continuous_action_threshold).astype(np.int32)
                + 1
            )
            vertical = (
                -(y < -self.continuous_action_threshold).astype(np.int32)
                + (y > self.continuous_action_threshold).astype(np.int32)
                + 1
            )
            fire = (actions[:, 2] > self.continuous_action_threshold).astype(np.int32)

            action_ids = self.map_action_idx[horizontal, vertical, fire]
            self.ale.send(action_ids, actions[:, 0])
        else:
            if isinstance(actions, (int, np.integer)):
                assert self.batch_size == 1, "A scalar action requires batch_size=1"
                actions = np.array([actions], dtype=np.int64)
            elif isinstance(actions, np.ndarray) and actions.ndim == 0:
                assert self.batch_size == 1, "A scalar action requires batch_size=1"
                actions = actions.reshape(1)
            elif isinstance(actions, (list, tuple)):
                assert (
                    len(actions) == self.batch_size
                ), f"Expected {self.batch_size} actions, got {len(actions)}"
                if len(actions) == 0 or not isinstance(actions[0], (int, np.integer)):
                    self.ale.send(
                        [np.asarray(a, dtype=np.int64).tolist() for a in actions],
                        paddle_strength,
                        gamma,
                    )
                    return
                actions = np.asarray(actions, dtype=np.int64)
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.int64 or actions.dtype == np.int32
            assert actions.shape == (
                self.batch_size,
            ), f"{actions.shape=}, {self.batch_size=}"

            if isinstance(paddle_strength, (int, float)):
                paddle_strength = np.full(
                    self.batch_size, paddle_strength, dtype=np.float32
                )
            else:
                paddle_strength = np.asarray(paddle_strength, dtype=np.float32)
            self.ale.send(actions, paddle_strength)

    def recv(
        self,
        obs_out: np.ndarray | None = None,
        reward_out: np.ndarray | None = None,
        term_out: np.ndarray | None = None,
        trunc_out: np.ndarray | None = None,
        steps_out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Receive next observations, rewards, terminations, truncations and info.

        When output buffers are provided, fills them in place and returns a tuple
        containing those same arrays. Without output buffers, new arrays are allocated.
        """
        result = self.ale.recv(obs_out, reward_out, term_out, trunc_out, steps_out)
        if isinstance(result, dict):
            return obs_out, reward_out, term_out, trunc_out, result
        return result

    def torch(self):
        """Register and return PyTorch custom ops for zero-copy ALE integration.

        Returns:
            (handle_id, ale_send, ale_step, ale_recv, unregister)
        """
        try:
            from ale_py._torch_ops import register_pytorch_ops
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                "ALE requires PyTorch for torch() support. Install with: pip install torch"
            ) from e
        return register_pytorch_ops(self)

    def xla(self):
        """Return XLA-compatible functions for JAX integration.

        Returns:
            (ale_handle, xla_reset, xla_step)
        """
        try:
            from ale_py._xla_ops import register_xla_ops
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                'ALE is missing jax, necessary for using the vector XLA support, use `pip install "ale_py[xla]"` to import'
            ) from e
        return register_xla_ops(self)
