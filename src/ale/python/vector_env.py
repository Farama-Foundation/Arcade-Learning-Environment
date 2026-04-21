"""Vector environment for ALE."""

from __future__ import annotations

from typing import Any

import ale_py
import gymnasium.vector.utils
import numpy as np
from ale_py import roms
from ale_py.env import AtariEnv
from gymnasium.core import ObsType
from gymnasium.spaces import Box, Discrete, MultiDiscrete
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
        _autoreset_str = autoreset_mode.value if isinstance(autoreset_mode, AutoresetMode) else autoreset_mode
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
                    f'{g} is not a ROM name, it should be snake_case not camel-case, '
                    f'i.e., "ms_pacman" not "MsPacman"'
                )
                rom_paths.append(rp)
            if num_envs is not None:
                assert num_envs == len(game), (
                    f"num_envs={num_envs} does not match len(game)={len(game)}"
                )
            num_envs = len(game)
            self.ale = ale_py.ALEVectorInterface(rom_paths=rom_paths, **_common)
        else:
            assert num_envs is not None, "num_envs is required when game is a str"
            rom_path = roms.get_rom_path(game)
            assert rom_path is not None, (
                f'{game} is not a ROM name, it should be snake_case not camel-case, '
                f'i.e., "ms_pacman" not "MsPacman"'
            )
            self.ale = ale_py.ALEVectorInterface(rom_path=rom_path, num_envs=num_envs, **_common)

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

        if self.continuous:
            # Actions are radius, theta, and fire, where first two are the parameters of polar coordinates.
            self.single_action_space = Box(
                low=np.array([0.0, -np.pi, 0.0]).astype(np.float32),
                high=np.array([1.0, np.pi, 1.0]).astype(np.float32),
                dtype=np.float32,
                shape=(3,),
            )
        elif len(set(self.num_actions)) == 1:
            # All envs have identical action counts (single ROM or same-action multi-ROM)
            self.single_action_space = Discrete(self.num_actions[0])
        else:
            # Heterogeneous action spaces - no meaningful single-env space
            self.single_action_space = None

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
            self.action_space = gymnasium.vector.utils.batch_space(
                self.single_action_space, self.batch_size
            )
        else:
            self.action_space = MultiDiscrete(
                np.array(self.num_actions[:self.batch_size], dtype=np.int64)
            )

        self.is_xla_registered = False

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
        self, actions: np.ndarray | list[np.ndarray], gamma: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Steps through the sub-environments.

        Pass a flat ndarray for a single action per env, or a list of ndarrays
        for variable-length sequences (gamma-discounted reward accumulation).
        """
        self.send(actions, gamma=gamma)
        return self.ale.recv()

    def send(self, actions: np.ndarray | list[np.ndarray], gamma: float = 1.0):
        """Send actions to the sub-environments.

        Pass a flat ndarray for a single action per env, or a list of ndarrays
        for variable-length sequences (gamma-discounted reward accumulation).
        """
        if isinstance(actions, list):
            assert len(actions) == self.batch_size, (
                f"Expected {self.batch_size} sequences, got {len(actions)}"
            )
            action_id_sequences = [a.tolist() if len(a) > 0 else [] for a in actions]
            paddle_strength_sequences = [[1.0] * len(a) for a in actions]
            self.ale.send_sequences(action_id_sequences, paddle_strength_sequences, gamma)
        elif self.continuous:
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
            paddle_strength = actions[:, 0]
            self.ale.send(action_ids, paddle_strength)
        else:
            assert isinstance(actions, np.ndarray)
            assert actions.dtype == np.int64 or actions.dtype == np.int32
            assert actions.shape == (
                self.batch_size,
            ), f"{actions.shape=}, {self.batch_size=}"

            paddle_strength = np.ones(self.batch_size)
            self.ale.send(actions, paddle_strength)

    def recv(
        self,
        obs_out: np.ndarray | None = None,
        reward_out: np.ndarray | None = None,
        term_out: np.ndarray | None = None,
        trunc_out: np.ndarray | None = None,
        steps_out: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | dict[str, Any]:
        """Receive the next step result.

        When output buffers are provided, fill them in place and return only the info dict.
        Otherwise return freshly allocated arrays and the info dict.
        """
        return self.ale.recv(obs_out, reward_out, term_out, trunc_out, steps_out)

    def torch(self):
        """Register and return PyTorch custom ops for zero-copy ALE integration.

        Similar to env.xla() for JAX - lazy-imports torch so it is not a hard dep.

        Returns:
            (handle_id, ale_send, ale_recv, ale_send_sequences, ale_send_sequences_nested,
             get_last_info, unregister)
        """
        try:
            from ale_py._torch_ops import register_pytorch_ops
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                "ALE requires PyTorch for torch() support. Install with: pip install torch"
            ) from e
        return register_pytorch_ops(self)

    def xla(self):
        """Return XLA-compatible functions for JAX integration."""
        try:
            import chex
            import jax
            import jax.numpy as jnp
        except ImportError as e:
            raise gymnasium.error.DependencyNotInstalled(
                'ALE is missing jax, necessary for using the vector XLA support, use `pip install "ale_py[xla]"` to import'
            ) from e

        if not self.is_xla_registered:
            # Register CPU targets
            jax.ffi.register_ffi_target(
                "atari_vector_xla_reset",
                ale_py._ale_py.VectorXLAReset(),
                platform="cpu",
            )
            jax.ffi.register_ffi_target(
                "atari_vector_xla_step", ale_py._ale_py.VectorXLAStep(), platform="cpu"
            )

            # Register GPU targets if available
            if hasattr(ale_py._ale_py, "VectorXLAResetGPU"):
                jax.ffi.register_ffi_target(
                    "atari_vector_xla_reset",
                    ale_py._ale_py.VectorXLAResetGPU(),
                    platform="CUDA",
                )
                jax.ffi.register_ffi_target(
                    "atari_vector_xla_step",
                    ale_py._ale_py.VectorXLAStepGPU(),
                    platform="CUDA",
                )

            self.is_xla_registered = True

        map_action_idx_jnp = jnp.array(self.map_action_idx)

        def xla_reset(
            handle: np.ndarray,
            seed: np.ndarray | None = None,
            reset_mask: np.ndarray | None = None,
        ) -> tuple[np.ndarray, tuple[np.ndarray, dict[str, np.ndarray]]]:
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
                has_side_effect=True,
            )

            if reset_mask is not None:
                reset_mask = jnp.asarray(reset_mask)

                chex.assert_shape(reset_mask, (self.num_envs,))
                chex.assert_type(reset_mask, jnp.bool_)

                (reset_indices,) = jnp.where(reset_mask)
                reset_indices = reset_indices.astype(jnp.int32)
            else:
                reset_indices = jnp.arange(self.num_envs, dtype=jnp.int32)

            if seed is None:
                reset_seeds = jnp.full(len(reset_indices), -1, dtype=jnp.int32)
            elif isinstance(seed, int):
                reset_seeds = jnp.arange(
                    seed, seed + len(reset_indices), dtype=jnp.int32
                )
            else:
                reset_seeds = jnp.asarray(seed, dtype=jnp.int32)

            chex.assert_shape(reset_seeds, (self.num_envs,))

            new_handle, obs, env_ids, lives, frame_numbers, episode_frame_numbers = (
                xla_call(handle, reset_indices, reset_seeds)
            )

            info = {
                "env_id": env_ids,
                "lives": lives,
                "frame_number": frame_numbers,
                "episode_frame_number": episode_frame_numbers,
            }
            return new_handle, (obs, info)

        def xla_step(handle, actions):
            # Convert to JAX array if needed (handles both numpy arrays and JAX tracers)
            actions = jnp.asarray(actions)

            if self.continuous:
                actions = actions.astype(jnp.float32)

                chex.assert_shape(actions, (self.batch_size, 3))
                chex.assert_type(actions, jnp.float32)

                x = actions[:, 0] * jnp.cos(actions[:, 1])
                y = actions[:, 0] * jnp.sin(actions[:, 1])

                horizontal = (
                    -(x < -self.continuous_action_threshold).astype(jnp.int32)
                    + (x > self.continuous_action_threshold).astype(jnp.int32)
                    + 1
                )
                vertical = (
                    -(y < -self.continuous_action_threshold).astype(jnp.int32)
                    + (y > self.continuous_action_threshold).astype(jnp.int32)
                    + 1
                )
                fire = (actions[:, 2] > self.continuous_action_threshold).astype(
                    jnp.int32
                )

                action_ids = map_action_idx_jnp[horizontal, vertical, fire]
                paddle_strength = actions[:, 0]
            else:
                action_ids = actions.astype(jnp.int32)
                paddle_strength = jnp.ones(self.batch_size, dtype=jnp.float32)

                chex.assert_shape(actions, (self.batch_size,))
                chex.assert_type(actions, jnp.int32)

            xla_call = jax.ffi.ffi_call(
                target_name="atari_vector_xla_step",
                result_shape_dtypes=(
                    jax.ShapeDtypeStruct((8,), jnp.uint8),  # handle
                    jax.ShapeDtypeStruct(  # observations
                        self.observation_space.shape, jnp.uint8
                    ),
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # rewards
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),  # terminations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.bool_),  # truncations
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # env_ids
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # lives
                    jax.ShapeDtypeStruct((self.num_envs,), jnp.int32),  # frame numbers
                    jax.ShapeDtypeStruct(  # episode frame number
                        (self.num_envs,), jnp.int32
                    ),
                ),
                vmap_method="broadcast_all",
                has_side_effect=True,
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
                "env_id": env_ids,
                "lives": lives,
                "frame_number": frame_numbers,
                "episode_frame_number": episode_frame_numbers,
            }
            return new_handle, (
                obs,
                rewards,
                terminations,
                truncations,
                info,
            )

        # Get the vectorizer handle and make sure it's properly formatted
        ale_handle = jnp.frombuffer(self.ale.handle(), dtype=np.uint8)
        return ale_handle, xla_reset, xla_step


def test_recv_with_optional_output_buffers() -> None:
    class _DummyALE:
        def __init__(self):
            self.recv_calls = 0

        def recv(self, obs_out=None, reward_out=None, term_out=None, trunc_out=None, steps_out=None):
            self.recv_calls += 1
            if obs_out is None:
                return (
                    np.zeros((2, 4, 84, 84), dtype=np.uint8),
                    np.zeros(2, dtype=np.float64),
                    np.zeros(2, dtype=bool),
                    np.zeros(2, dtype=bool),
                    {"mode": "recv"},
                )
            obs_out.fill(1)
            reward_out.fill(2)
            term_out.fill(False)
            trunc_out.fill(False)
            steps_out.fill(3)
            return {"mode": "recv_into"}

    env = AtariVectorEnv.__new__(AtariVectorEnv)
    env.ale = _DummyALE()

    obs, rewards, terms, truncs, info = env.recv()
    assert obs.shape == (2, 4, 84, 84)
    assert rewards.shape == (2,)
    assert not terms.any()
    assert not truncs.any()
    assert info == {"mode": "recv"}
    assert env.ale.recv_calls == 1

    obs_out = np.empty((2, 4, 84, 84), dtype=np.uint8)
    reward_out = np.empty(2, dtype=np.float64)
    term_out = np.empty(2, dtype=bool)
    trunc_out = np.empty(2, dtype=bool)
    steps_out = np.empty(2, dtype=np.int32)
    info = env.recv(obs_out, reward_out, term_out, trunc_out, steps_out)
    assert info == {"mode": "recv_into"}
    assert env.ale.recv_calls == 2
    assert (obs_out == 1).all()
    assert (reward_out == 2).all()
    assert (steps_out == 3).all()
