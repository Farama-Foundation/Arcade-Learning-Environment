"""XLA/JAX custom ops for ALE - registered via AtariVectorEnv.xla().

Do not import this module at the top level; it is lazy-loaded by xla().
"""

import ale_py
import chex
import jax
import jax.numpy as jnp
import numpy as np

_xla_registered: bool = False


def register_xla_ops(env):
    """Register XLA targets once and return (ale_handle, xla_reset, xla_step)."""
    global _xla_registered

    if not _xla_registered:
        jax.ffi.register_ffi_target(
            "atari_vector_xla_reset",
            ale_py._ale_py.VectorXLAReset(),
            platform="cpu",
        )
        jax.ffi.register_ffi_target(
            "atari_vector_xla_step", ale_py._ale_py.VectorXLAStep(), platform="cpu"
        )

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

        _xla_registered = True

    map_action_idx_jnp = jnp.array(env.map_action_idx)

    def xla_reset(
        handle: np.ndarray,
        seed: np.ndarray | None = None,
        reset_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, dict[str, np.ndarray]]]:
        xla_call = jax.ffi.ffi_call(
            target_name="atari_vector_xla_reset",
            result_shape_dtypes=(
                jax.ShapeDtypeStruct((8,), jnp.uint8),
                jax.ShapeDtypeStruct(env.observation_space.shape, jnp.uint8),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
            ),
            vmap_method="broadcast_all",
            has_side_effect=True,
        )

        if reset_mask is not None:
            reset_mask = jnp.asarray(reset_mask)
            chex.assert_shape(reset_mask, (env.num_envs,))
            chex.assert_type(reset_mask, jnp.bool_)
            (reset_indices,) = jnp.where(reset_mask)
            reset_indices = reset_indices.astype(jnp.int32)
        else:
            reset_indices = jnp.arange(env.num_envs, dtype=jnp.int32)

        if seed is None:
            reset_seeds = jnp.full(len(reset_indices), -1, dtype=jnp.int32)
        elif isinstance(seed, int):
            reset_seeds = jnp.arange(seed, seed + len(reset_indices), dtype=jnp.int32)
        else:
            reset_seeds = jnp.asarray(seed, dtype=jnp.int32)

        chex.assert_shape(reset_seeds, (env.num_envs,))

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
        actions = jnp.asarray(actions)

        if env.continuous:
            actions = actions.astype(jnp.float32)
            chex.assert_shape(actions, (env.batch_size, 3))
            chex.assert_type(actions, jnp.float32)

            x = actions[:, 0] * jnp.cos(actions[:, 1])
            y = actions[:, 0] * jnp.sin(actions[:, 1])

            horizontal = (
                -(x < -env.continuous_action_threshold).astype(jnp.int32)
                + (x > env.continuous_action_threshold).astype(jnp.int32)
                + 1
            )
            vertical = (
                -(y < -env.continuous_action_threshold).astype(jnp.int32)
                + (y > env.continuous_action_threshold).astype(jnp.int32)
                + 1
            )
            fire = (actions[:, 2] > env.continuous_action_threshold).astype(jnp.int32)

            action_ids = map_action_idx_jnp[horizontal, vertical, fire]
            paddle_strength = actions[:, 0]
        else:
            action_ids = actions.astype(jnp.int32)
            paddle_strength = jnp.ones(env.batch_size, dtype=jnp.float32)
            chex.assert_shape(actions, (env.batch_size,))
            chex.assert_type(actions, jnp.int32)

        xla_call = jax.ffi.ffi_call(
            target_name="atari_vector_xla_step",
            result_shape_dtypes=(
                jax.ShapeDtypeStruct((8,), jnp.uint8),
                jax.ShapeDtypeStruct(env.observation_space.shape, jnp.uint8),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.bool_),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.bool_),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
                jax.ShapeDtypeStruct((env.num_envs,), jnp.int32),
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
        return new_handle, (obs, rewards, terminations, truncations, info)

    ale_handle = jnp.frombuffer(env.ale.handle(), dtype=np.uint8)
    return ale_handle, xla_reset, xla_step
