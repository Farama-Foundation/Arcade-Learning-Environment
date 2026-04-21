"""PyTorch custom ops for ALE - registered via AtariVectorEnv.torch().

No 'from __future__ import annotations' here so torch.library.custom_op can
infer op schemas from type annotations without PEP 563 string-ification.

Do not import this module at the top level; it is lazy-loaded by torch().
"""

import numpy as np
import torch
from torch._library.effects import EffectType
from typing import Any

# Module-level state shared across all registered envs
_torch_registered: bool = False
_torch_envs: dict[int, Any] = {}
_torch_buffers: dict[int, dict] = {}
_torch_last_info: dict[int, dict] = {}


def register_pytorch_ops(env: Any):
    """Allocate pinned buffers for env and register ale::* torch custom ops once.

    Returns:
        (handle_id, ale_send, ale_recv, ale_send_sequences,
         ale_send_sequences_nested, get_last_info, unregister)
    """
    global _torch_registered

    handle_id = id(env)
    num_envs = env.num_envs
    obs_shape = (num_envs,) + tuple(env.observation_space.shape[1:])

    _torch_envs[handle_id] = env
    _torch_buffers[handle_id] = {
        "actions": torch.empty(num_envs, dtype=torch.int64, pin_memory=True),
        "obs": torch.empty(obs_shape, dtype=torch.uint8, pin_memory=True),
        "reward": torch.empty(num_envs, dtype=torch.float64, pin_memory=True),
        "term": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "trunc": torch.empty(num_envs, dtype=torch.bool, pin_memory=True),
        "steps_taken": torch.empty(num_envs, dtype=torch.int32, pin_memory=True),
    }

    if not _torch_registered:
        _torch_registered = True

        @torch.library.custom_op("ale::send", mutates_args=())
        def ale_send(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            buf["actions"].copy_(actions.detach(), non_blocking=True)
            _torch_envs[handle_id].send(buf["actions"].numpy())
            return actions.new_empty(())

        ale_send.register_effect(EffectType.ORDERED)

        @ale_send.register_fake
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty(())

        @torch.library.custom_op("ale::recv", mutates_args=())
        def ale_recv(handle_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            buf = _torch_buffers[handle_id]
            info = _torch_envs[handle_id].recv(
                buf["obs"].numpy(),
                buf["reward"].numpy(),
                buf["term"].numpy(),
                buf["trunc"].numpy(),
                buf["steps_taken"].numpy(),
            )
            _torch_last_info[handle_id] = info
            return (buf["obs"], buf["reward"], buf["term"], buf["trunc"], buf["steps_taken"])

        ale_recv.register_effect(EffectType.ORDERED)

        @ale_recv.register_fake
        def _(handle_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            buf = _torch_buffers[handle_id]
            return (
                torch.empty(buf["obs"].shape, dtype=buf["obs"].dtype),
                torch.empty(buf["reward"].shape, dtype=buf["reward"].dtype),
                torch.empty(buf["term"].shape, dtype=buf["term"].dtype),
                torch.empty(buf["trunc"].shape, dtype=buf["trunc"].dtype),
                torch.empty(buf["steps_taken"].shape, dtype=buf["steps_taken"].dtype),
            )

        @torch.library.custom_op("ale::send_sequences", mutates_args=())
        def ale_send_sequences(
            handle_id: int, action_ids: torch.Tensor, lengths: torch.Tensor, gamma: float
        ) -> torch.Tensor:
            action_ids_np = action_ids.detach().cpu().numpy().astype(np.int64)
            lengths_np = lengths.detach().cpu().numpy().astype(np.int64)
            seqs = [action_ids_np[i, : lengths_np[i]].copy() for i in range(len(lengths_np))]
            _torch_envs[handle_id].send(seqs, gamma=gamma)
            return action_ids.new_empty(())

        ale_send_sequences.register_effect(EffectType.ORDERED)

        @ale_send_sequences.register_fake
        def _(handle_id: int, action_ids: torch.Tensor, lengths: torch.Tensor, gamma: float) -> torch.Tensor:
            return action_ids.new_empty(())

        @torch.library.custom_op("ale::send_sequences_nested", mutates_args=())
        def ale_send_sequences_nested(
            handle_id: int, values: torch.Tensor, offsets: torch.Tensor, gamma: float
        ) -> torch.Tensor:
            values_np = values.detach().cpu().numpy().astype(np.int64)
            offsets_np = offsets.detach().cpu().numpy().astype(np.int64)
            seqs = [values_np[offsets_np[i] : offsets_np[i + 1]].copy() for i in range(len(offsets_np) - 1)]
            _torch_envs[handle_id].send(seqs, gamma=gamma)
            return values.new_empty(())

        ale_send_sequences_nested.register_effect(EffectType.ORDERED)

        @ale_send_sequences_nested.register_fake
        def _(handle_id: int, values: torch.Tensor, offsets: torch.Tensor, gamma: float) -> torch.Tensor:
            return values.new_empty(())

    def get_last_info() -> dict:
        return _torch_last_info.get(handle_id, {})

    def unregister() -> None:
        _torch_envs.pop(handle_id, None)
        _torch_buffers.pop(handle_id, None)
        _torch_last_info.pop(handle_id, None)

    return (
        handle_id,
        torch.ops.ale.send,
        torch.ops.ale.recv,
        torch.ops.ale.send_sequences,
        torch.ops.ale.send_sequences_nested,
        get_last_info,
        unregister,
    )


class TorchOpsWrapper:
    """Wraps an AtariVectorEnv with torch custom ops for compiled training.

    Exposes ale_send, ale_recv, ale_send_sequences, ale_send_sequences_nested
    as bound methods and tracks handle_id for use as the first argument.
    """

    def __init__(self, env):
        self._env = env
        (
            self.handle_id,
            self.ale_send,
            self.ale_recv,
            self.ale_send_sequences,
            self.ale_send_sequences_nested,
            self._get_last_info,
            self._unregister,
        ) = env.torch()

    def get_last_info(self) -> dict:
        return self._get_last_info()

    def close(self):
        self._unregister()
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name: str):
        return getattr(self._env, name)
