"""PyTorch custom ops for ALE - registered via AtariVectorEnv.torch().

No 'from __future__ import annotations' here so torch.library.custom_op can
infer op schemas from type annotations without PEP 563 string-ification.

Do not import this module at the top level; it is lazy-loaded by torch().
"""

from typing import Any

import torch
from torch._library.effects import EffectType

__all__ = ["TorchOpsWrapper", "register_pytorch_ops"]

# Module-level state shared across all registered envs.
# _torch_registered is never reset: torch.library.custom_op raises if the same name is
# registered twice, so ops are registered once for the lifetime of the process.
_torch_registered: bool = False
_torch_envs: dict[int, Any] = {}
_torch_buffers: dict[int, dict] = {}
_torch_last_info: dict[int, dict] = {}


def register_pytorch_ops(env: Any):
    """Allocate pinned buffers for env and register ale::* torch custom ops once.

    Returns:
        (handle_id, ale_send, ale_step, ale_recv, get_last_info, unregister)
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
        "actions_d2h_event": torch.cuda.Event() if torch.cuda.is_available() else None,
        "sequences_d2h_event": torch.cuda.Event() if torch.cuda.is_available() else None,
    }

    if not _torch_registered:
        _torch_registered = True

        @torch.library.custom_op("ale::send", mutates_args=())
        @torch.no_grad()
        def ale_send(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            buf["actions"].copy_(actions, non_blocking=True)
            if actions.is_cuda:
                buf["actions_d2h_event"].record()
                buf["actions_d2h_event"].synchronize()
            _torch_envs[handle_id].send(buf["actions"].numpy())
            return actions.new_empty(())

        ale_send.register_effect(EffectType.ORDERED)

        @ale_send.register_fake
        def _(handle_id: int, actions: torch.Tensor) -> torch.Tensor:
            return actions.new_empty(())

        @torch.library.custom_op("ale::recv", mutates_args=())
        def ale_recv(
            handle_id: int,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            buf = _torch_buffers[handle_id]
            info = _torch_envs[handle_id].recv(
                buf["obs"].numpy(),
                buf["reward"].numpy(),
                buf["term"].numpy(),
                buf["trunc"].numpy(),
                buf["steps_taken"].numpy(),
            )
            _torch_last_info[handle_id] = info
            return (
                buf["obs"],
                buf["reward"],
                buf["term"],
                buf["trunc"],
                buf["steps_taken"],
            )

        ale_recv.register_effect(EffectType.ORDERED)

        @ale_recv.register_fake
        def _(
            handle_id: int,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ]:
            buf = _torch_buffers[handle_id]
            return (
                torch.empty(buf["obs"].shape, dtype=buf["obs"].dtype),
                torch.empty(buf["reward"].shape, dtype=buf["reward"].dtype),
                torch.empty(buf["term"].shape, dtype=buf["term"].dtype),
                torch.empty(buf["trunc"].shape, dtype=buf["trunc"].dtype),
                torch.empty(buf["steps_taken"].shape, dtype=buf["steps_taken"].dtype),
            )

        @torch.library.custom_op("ale::send_sequences", mutates_args=())
        @torch.no_grad()
        def ale_send_sequences(
            handle_id: int, values: torch.Tensor, offsets: torch.Tensor, gamma: float
        ) -> torch.Tensor:
            buf = _torch_buffers[handle_id]
            offsets_cpu = offsets.to("cpu", non_blocking=True)
            values_cpu = values.to("cpu", non_blocking=True)
            if offsets.is_cuda or values.is_cuda:
                buf["sequences_d2h_event"].record()
                buf["sequences_d2h_event"].synchronize()
            offsets_np = offsets_cpu.numpy()
            values_np = values_cpu.numpy()
            action_sequences = [
                values_np[offsets_np[i] : offsets_np[i + 1]]
                for i in range(len(offsets_np) - 1)
            ]
            _torch_envs[handle_id].send(action_sequences, gamma=gamma)
            return values.new_empty(())

        ale_send_sequences.register_effect(EffectType.ORDERED)

        @ale_send_sequences.register_fake
        def _(
            handle_id: int, values: torch.Tensor, offsets: torch.Tensor, gamma: float
        ) -> torch.Tensor:
            return values.new_empty(())

    def get_last_info() -> dict:
        return _torch_last_info.get(handle_id, {})

    def unregister() -> None:
        _torch_envs.pop(handle_id, None)
        _torch_buffers.pop(handle_id, None)
        _torch_last_info.pop(handle_id, None)

    def ale_step(
        handle_id: int, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.ops.ale.send(handle_id, actions)
        return torch.ops.ale.recv(handle_id)

    return (
        handle_id,
        torch.ops.ale.send,
        ale_step,
        torch.ops.ale.recv,
        get_last_info,
        unregister,
    )


class TorchOpsWrapper:
    """Wraps an AtariVectorEnv with torch custom ops for compiled training.

    Exposes ale_send, ale_step, and ale_recv as bound methods and tracks handle_id
    for use as the first argument.
    """

    def __init__(self, env):
        """Initialize by calling env.torch() and binding the returned ops."""
        self._env = env
        (
            self.handle_id,
            self.ale_send,
            self.ale_step,
            self.ale_recv,
            self._get_last_info,
            self._unregister,
        ) = env.torch()

    def get_last_info(self) -> dict:
        """Return info dict from the most recent recv call."""
        return self._get_last_info()

    def close(self):
        """Unregister torch buffers and close the underlying env."""
        self._unregister()
        if hasattr(self._env, "close"):
            self._env.close()

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped env."""
        return getattr(self._env, name)
