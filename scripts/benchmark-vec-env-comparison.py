#!/usr/bin/env python3
r"""Cross-library Atari vectorisation benchmark: ALE vs EnvPool, PufferLib and Gymnasium.

Libraries with a native `send`/`recv` API (ALE, EnvPool) can be much faster through it
than through a synchronous `step()` wrapper, so a comparison that drives every backend
through `step()` can misrank them. This script measures each backend on every API it
supports, with the same preprocessing, so the comparison is like for like.

Only `ale` is required; the other backends are imported when they are first used, and a
missing package is reported with the name to install.

Backends and the APIs each supports:

| backend      | `--api step` | `--api send-recv` | parallelism |
|--------------|--------------|-------------------|-------------|
| `ale`        | yes          | yes (`send`/`recv`) | `num_threads` |
| `envpool`    | yes          | yes (`send`/`recv`) | `num_threads` |
| `pufferlib`  | yes (send+recv) | yes             | `num_workers` processes |
| `gymnasium-sync`  | yes     | no                | single process |
| `gymnasium-async` | yes     | no                | one process per env |

ALE's `step()` is literally `send(); recv()`, so both of its rows should agree to within
noise; that agreement is itself the check that the harness is not adding cost. Any
EnvPool `step()` vs `send`/`recv` gap is then the finding.

All backends are configured to do the *same work*: 84x84 grayscale, frameskip 4, maxpool
over the last two frames, 4-frame stack, noop-30 and fire reset, reward clipping,
sticky actions off, 108,000-frame (27,000-step) episode cap. `--verify-work` (on by
default where env ids are available) records the per-env `recv` histogram so two APIs can
be shown to have emulated the same number of steps before their throughputs are compared.

Examples:
    # ALE and EnvPool, both APIs, across the full env range
    ./scripts/benchmark-vec-env-comparison.py --backends ale,envpool --api both \\
        --num-envs 8,32,64,128,256,512 --repeats 5

    # Every backend on the synchronous API
    ./scripts/benchmark-vec-env-comparison.py --backends all --api step

    # Async comparison, worker counts pinned on both sides
    ./scripts/benchmark-vec-env-comparison.py --backends ale,envpool --api send-recv \\
        --num-envs 256,512 --batch-divisors 2,3 --num-threads 24
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _benchmark_common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    MeasuredRun,
    ResultStore,
    emit_result,
    iter_int_list,
    measure,
    summarise,
    sweep,
)

SCRIPT = Path(__file__).resolve()

# Package to install for each optional backend.
OPTIONAL_PACKAGES = {"envpool": "envpool", "pufferlib": "pufferlib"}

BACKENDS = (
    "ale",
    "envpool",
    "pufferlib",
    "gymnasium-sync",
    "gymnasium-async",
)
NATIVE_API_BACKENDS = ("ale", "envpool", "pufferlib")

# Shared task definition. Every backend must be configured to match this exactly;
# changing anything here invalidates comparisons against previously stored results.
TASK = {
    "img_size": 84,
    "grayscale": True,
    "frameskip": 4,
    "stack_num": 4,
    "maxpool": True,
    "noop_max": 30,
    "use_fire_reset": True,
    "episodic_life": False,
    "reward_clipping": True,
    "repeat_action_probability": 0.0,
    "full_action_space": False,
    "max_num_frames_per_episode": 108_000,
}
MAX_EPISODE_STEPS = TASK["max_num_frames_per_episode"] // TASK["frameskip"]


def camel_case(game: str) -> str:
    """Convert a snake_case ROM name to the CamelCase form used in env ids."""
    return "".join(part.capitalize() for part in game.split("_"))


# --------------------------------------------------------------------------------------
# ALE
# --------------------------------------------------------------------------------------


class ALEBackend:
    """ALE's `AtariVectorEnv`, driven through `step()` or native `send`/`recv`."""

    def __init__(self, config: dict[str, Any]):
        """Store the configuration; the environment is built by `build()`."""
        self.config = config
        self.name = "ale"
        self.num_envs = int(config["num_envs"])
        self.batch_size = int(config["batch_size"]) or self.num_envs
        self.envs: Any = None
        self._send_recv = config["api"] == "send-recv"

    def build(self) -> None:
        """Construct the vector environment."""
        from ale_py import AtariVectorEnv

        self.envs = AtariVectorEnv(
            game=self.config["game"],
            num_envs=self.num_envs,
            batch_size=0 if self.batch_size == self.num_envs else self.batch_size,
            num_threads=int(self.config["workers"]),
            img_height=TASK["img_size"],
            img_width=TASK["img_size"],
            grayscale=TASK["grayscale"],
            frameskip=TASK["frameskip"],
            stack_num=TASK["stack_num"],
            maxpool=TASK["maxpool"],
            noop_max=TASK["noop_max"],
            use_fire_reset=TASK["use_fire_reset"],
            episodic_life=TASK["episodic_life"],
            reward_clipping=TASK["reward_clipping"],
            repeat_action_probability=TASK["repeat_action_probability"],
            full_action_space=TASK["full_action_space"],
            max_num_frames_per_episode=TASK["max_num_frames_per_episode"],
        )

    def reset(self) -> None:
        """Reset every sub-environment."""
        self.envs.reset(seed=0)

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> np.ndarray:
        """Return pre-generated action batches."""
        num_actions = self.envs.single_action_space.n
        return rng.integers(
            0, num_actions, size=(pool_size, self.batch_size), dtype=np.int32
        )

    def step_batch(self, actions: np.ndarray) -> np.ndarray:
        """Advance one batch and return the received env ids."""
        if self._send_recv:
            self.envs.send(actions)
            return self.envs.recv()[4]["env_id"]
        return self.envs.step(actions)[4]["env_id"]

    def close(self) -> None:
        """Close the vector environment."""
        if self.envs is not None:
            self.envs.close()
            self.envs = None


# --------------------------------------------------------------------------------------
# EnvPool
# --------------------------------------------------------------------------------------


class EnvPoolBackend:
    """EnvPool's Atari pool, driven through `step()` or native `send`/`recv`.

    In the `send`/`recv` path the loop is seeded by `async_reset()` plus one `recv()`,
    then alternates `send(actions, last_env_ids)` and `recv()`. This is EnvPool's own
    benchmark protocol.
    """

    def __init__(self, config: dict[str, Any]):
        """Store the configuration; the pool is built by `build()`."""
        self.config = config
        self.name = "envpool"
        self.num_envs = int(config["num_envs"])
        self.batch_size = int(config["batch_size"]) or self.num_envs
        self.envs: Any = None
        self._send_recv = config["api"] == "send-recv"
        self._last_env_ids: np.ndarray | None = None

    def build(self) -> None:
        """Construct the pool with settings matching `TASK`."""
        import envpool

        self.envs = envpool.make(
            f"{camel_case(self.config['game'])}-v5",
            env_type="gymnasium",
            num_envs=self.num_envs,
            batch_size=self.batch_size,
            num_threads=int(self.config["workers"]),
            seed=0,
            img_height=TASK["img_size"],
            img_width=TASK["img_size"],
            gray_scale=TASK["grayscale"],
            frame_skip=TASK["frameskip"],
            stack_num=TASK["stack_num"],
            noop_max=TASK["noop_max"],
            use_fire_reset=TASK["use_fire_reset"],
            episodic_life=TASK["episodic_life"],
            reward_clip=TASK["reward_clipping"],
            repeat_action_probability=TASK["repeat_action_probability"],
            full_action_space=TASK["full_action_space"],
            max_episode_steps=MAX_EPISODE_STEPS,
            use_inter_area_resize=bool(self.config.get("inter_area_resize", True)),
        )

    def reset(self) -> None:
        """Reset the pool, seeding the `send`/`recv` loop if that API is in use."""
        if self._send_recv:
            self.envs.async_reset()
            self._last_env_ids = self.envs.recv()[-1]["env_id"]
        else:
            self.envs.reset()

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> np.ndarray:
        """Return pre-generated action batches."""
        num_actions = self.envs.action_space.n
        return rng.integers(
            0, num_actions, size=(pool_size, self.batch_size), dtype=np.int32
        )

    def step_batch(self, actions: np.ndarray) -> np.ndarray:
        """Advance one batch and return the received env ids."""
        if self._send_recv:
            self.envs.send(actions, self._last_env_ids)
            env_ids = self.envs.recv()[-1]["env_id"]
            self._last_env_ids = env_ids
            return env_ids
        return self.envs.step(actions)[-1]["env_id"]

    def close(self) -> None:
        """Release the pool."""
        self.envs = None


# --------------------------------------------------------------------------------------
# Gymnasium-based backends
# --------------------------------------------------------------------------------------


def make_gymnasium_env(game: str):
    """Build a single Gymnasium ALE env preprocessed to match `TASK`.

    Args:
        game: ROM name in snake_case.

    Returns:
        A wrapped `gymnasium.Env` producing 84x84 grayscale 4-stacks with clipped rewards.
    """
    import ale_py
    import gymnasium as gym
    from gymnasium.wrappers import (
        AtariPreprocessing,
        FrameStackObservation,
        TransformReward,
    )

    gym.register_envs(ale_py)
    env = gym.make(
        f"ALE/{camel_case(game)}-v5",
        frameskip=1,
        repeat_action_probability=TASK["repeat_action_probability"],
        max_num_frames_per_episode=TASK["max_num_frames_per_episode"],
    )
    env = AtariPreprocessing(
        env,
        noop_max=TASK["noop_max"],
        frame_skip=TASK["frameskip"],
        screen_size=TASK["img_size"],
        grayscale_obs=TASK["grayscale"],
    )
    env = TransformReward(env, lambda reward: float(np.sign(reward)))
    return FrameStackObservation(env, TASK["stack_num"])


class GymnasiumBackend:
    """Gymnasium's `SyncVectorEnv` / `AsyncVectorEnv` over the same wrapped env."""

    def __init__(self, config: dict[str, Any]):
        """Store the configuration; the vector env is built by `build()`."""
        self.config = config
        self.name = config["backend"]
        self.num_envs = int(config["num_envs"])
        self.batch_size = self.num_envs
        self.envs: Any = None
        self._env_ids = np.arange(self.num_envs)

    def build(self) -> None:
        """Construct the vector environment."""
        import gymnasium as gym

        creator = functools.partial(make_gymnasium_env, self.config["game"])
        creators = [creator] * self.num_envs
        if self.name == "gymnasium-async":
            self.envs = gym.vector.AsyncVectorEnv(creators)
        else:
            self.envs = gym.vector.SyncVectorEnv(creators)

    def reset(self) -> None:
        """Reset every sub-environment."""
        self.envs.reset(seed=0)

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> np.ndarray:
        """Return pre-generated action batches."""
        num_actions = self.envs.single_action_space.n
        return rng.integers(
            0, num_actions, size=(pool_size, self.num_envs), dtype=np.int64
        )

    def step_batch(self, actions: np.ndarray) -> np.ndarray:
        """Advance every sub-environment once."""
        self.envs.step(actions)
        return self._env_ids

    def close(self) -> None:
        """Close the vector environment and its workers."""
        if self.envs is not None:
            self.envs.close()
            self.envs = None


# --------------------------------------------------------------------------------------
# PufferLib
# --------------------------------------------------------------------------------------


def puffer_creator(game: str, buf=None, seed: int = 0):
    """Return a PufferLib-wrapped copy of the shared Gymnasium env.

    PufferLib's shipped Atari creator uses different preprocessing (105x80, no maxpool,
    no noop/fire reset, no reward clipping, stack 1), so the shared Gymnasium env is
    used instead to keep the work identical across backends. This measures PufferLib's
    vectorisation backend rather than out-of-the-box PufferLib.

    Args:
        game: ROM name in snake_case.
        buf: Shared-memory buffers supplied by PufferLib.
        seed: Per-env seed supplied by PufferLib.

    Returns:
        A `GymnasiumPufferEnv` wrapping the shared env.
    """
    from pufferlib.emulation import GymnasiumPufferEnv

    return GymnasiumPufferEnv(env=make_gymnasium_env(game), buf=buf, seed=seed)


class PufferLibBackend:
    """PufferLib's multiprocessing vectorisation over the shared Gymnasium env."""

    def __init__(self, config: dict[str, Any]):
        """Store the configuration; the vector env is built by `build()`."""
        self.config = config
        self.name = "pufferlib"
        self.num_envs = int(config["num_envs"])
        self.batch_size = int(config["batch_size"]) or self.num_envs
        self.envs: Any = None
        self.extra: dict[str, Any] = {}
        self._env_ids: np.ndarray | None = None

    def build(self) -> None:
        """Construct the vector env, choosing a worker count that divides `num_envs`."""
        import pufferlib.vector

        workers = int(self.config["workers"])
        while workers > 1 and (
            self.num_envs % workers or self.batch_size % (self.num_envs // workers)
        ):
            workers -= 1

        self.envs = pufferlib.vector.make(
            functools.partial(puffer_creator, self.config["game"]),
            num_envs=self.num_envs,
            backend=pufferlib.vector.Multiprocessing,
            num_workers=workers,
            batch_size=self.batch_size,
            overwork=True,
        )
        # Recorded in metadata rather than folded into `config`: the config dict is the
        # resume key, so mutating it would make every stored PufferLib row un-resumable.
        self.extra = {"resolved_workers": workers}

    def reset(self) -> None:
        """Async-reset and take the first batch, seeding the send/recv loop."""
        self.envs.async_reset(seed=0)
        self._env_ids = np.asarray(self.envs.recv()[5])

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> np.ndarray:
        """Return pre-generated action batches."""
        num_actions = self.envs.single_action_space.n
        return rng.integers(
            0, num_actions, size=(pool_size, self.batch_size), dtype=np.int32
        )

    def step_batch(self, actions: np.ndarray) -> np.ndarray:
        """Advance one batch and return the received env ids."""
        self.envs.send(actions)
        self._env_ids = np.asarray(self.envs.recv()[5])
        return self._env_ids

    def close(self) -> None:
        """Close the vector env and its worker processes."""
        if self.envs is not None:
            self.envs.close()
            self.envs = None


BACKEND_CLASSES: dict[str, Any] = {
    "ale": ALEBackend,
    "envpool": EnvPoolBackend,
    "pufferlib": PufferLibBackend,
    "gymnasium-sync": GymnasiumBackend,
    "gymnasium-async": GymnasiumBackend,
}


def run_one(config: dict[str, Any], args: Namespace) -> MeasuredRun:
    """Measure a single configuration in this process."""
    backend = BACKEND_CLASSES[config["backend"]](config)
    return measure(
        backend,
        config,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        frameskip=TASK["frameskip"],
        seed=int(config.get("repeat", 0)),
        record_latency=not args.no_latency,
        verify_work=not args.no_verify_work,
    )


def build_configs(args: Namespace) -> list[dict[str, Any]]:
    """Expand the CLI arguments into the list of configurations to measure."""
    backends = (
        list(BACKENDS) if args.backends == ["all"] else [b for b in args.backends]
    )
    unknown = [b for b in backends if b not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s): {', '.join(unknown)}")
    apis = ["step", "send-recv"] if args.api == "both" else [args.api]

    configs: list[dict[str, Any]] = []
    for backend in backends:
        # PufferLib has one protocol: its step() is send+recv over the same buffers, so
        # it contributes a single "native" row however many APIs were requested.
        # Gymnasium's vector API only expresses step().
        if backend == "pufferlib":
            backend_apis = ["native"]
        elif backend in NATIVE_API_BACKENDS:
            backend_apis = apis
        else:
            backend_apis = [api for api in apis if api == "step"]

        for api in backend_apis:
            for num_envs in args.num_envs:
                for divisor in args.batch_divisors:
                    batch_size = num_envs // divisor
                    if batch_size < 1:
                        continue
                    if divisor != 1 and backend.startswith("gymnasium"):
                        continue  # Gymnasium's vector API cannot express async batching
                    configs.append(
                        {
                            "backend": backend,
                            "api": api,
                            "game": args.game,
                            "num_envs": num_envs,
                            "batch_size": batch_size,
                            "workers": min(args.num_threads, num_envs),
                        }
                    )
    return configs


def check_backends_installed(backends: set[str]) -> None:
    """Exit with an install hint if an optional backend's package is missing."""
    import importlib.util

    missing = sorted(
        package
        for backend, package in OPTIONAL_PACKAGES.items()
        if backend in backends and importlib.util.find_spec(package) is None
    )
    if missing:
        raise SystemExit(
            f"missing optional backend package(s): {', '.join(missing)}. "
            f"Install with `pip install {' '.join(missing)}` or drop them from --backends."
        )


def parse_args() -> Namespace:
    """Parse the command line."""
    parser = ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--backends",
        type=lambda s: s.split(","),
        default=["ale", "envpool"],
        help=f"comma separated: all,{','.join(BACKENDS)}",
    )
    parser.add_argument("--api", default="both", choices=["step", "send-recv", "both"])
    parser.add_argument("--game", default="breakout", help="ROM name (snake_case)")
    parser.add_argument(
        "--num-envs",
        type=lambda s: list(iter_int_list(s)),
        default=[8, 32, 64, 128, 256, 512],
    )
    parser.add_argument(
        "--batch-divisors",
        type=lambda s: list(iter_int_list(s)),
        default=[1],
        help="batch_size = num_envs // divisor; 1 is synchronous",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=psutil.cpu_count(logical=True),
        help="thread/worker cap, applied identically to every backend",
    )
    parser.add_argument("--warmup-seconds", type=float, default=8.0)
    parser.add_argument("--measure-seconds", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--out", help="JSONL results path (default: benchmark-results/comparison.jsonl)"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="per-run wall-clock limit (process backends can hang on construction)",
    )
    parser.add_argument("--no-isolate", action="store_true")
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--no-verify-work", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the configurations that would run and exit",
    )
    parser.add_argument(
        "--summary", action="store_true", help="summarise the existing store and exit"
    )
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Entry point: run one configuration, or drive the whole comparison."""
    args = parse_args()

    if args.run_one:
        emit_result(run_one(json.loads(args.run_one), args))
        return

    store = ResultStore(
        Path(args.out) if args.out else DEFAULT_RESULTS_DIR / "comparison.jsonl"
    )
    group_keys = ["backend", "api", "game", "num_envs", "batch_size"]

    if args.summary:
        summarise(store, group_keys)
        return

    configs = build_configs(args)
    check_backends_installed({config["backend"] for config in configs})
    print(f"{len(configs)} configuration(s) x {args.repeats} repeat(s)")
    estimate = (
        len(configs) * args.repeats * (args.warmup_seconds + args.measure_seconds + 30)
    )
    print(f"rough wall-clock estimate: {estimate / 60:.0f} min")

    if args.dry_run:
        for config in configs:
            print("  " + ", ".join(f"{k}={v}" for k, v in config.items()))
        return

    extra_args = [
        "--warmup-seconds",
        str(args.warmup_seconds),
        "--measure-seconds",
        str(args.measure_seconds),
    ]
    if args.no_latency:
        extra_args.append("--no-latency")
    if args.no_verify_work:
        extra_args.append("--no-verify-work")

    sweep(
        SCRIPT,
        configs,
        store,
        repeats=args.repeats,
        isolate=not args.no_isolate,
        extra_args=extra_args,
        inline_runner=lambda config: run_one(config, args),
        timeout=args.timeout,
    )

    print()
    summarise(store, group_keys)


if __name__ == "__main__":
    main()
