#!/usr/bin/env python3
"""Throughput and startup benchmarks for `AtariVectorEnv`.

| suite       | what it varies |
|-------------|----------------|
| `quick`     | sync, 16 and 64 envs, short windows; a smoke test (default) |
| `scaling`   | sync throughput across 8..512 envs |
| `async`     | `batch_size < num_envs` at several ratios, threads pinned |
| `threads`   | worker count, spanning the SMT boundary |
| `reset`     | reset frequency alone (episode cap), sync and async at each cap |
| `autoreset` | `NextStep` vs `SameStep` |
| `obs`       | grayscale vs RGB observations |
| `games`     | several ROMs, since one game does not generalise |
| `startup`   | constructor and first-reset cost |

Every run is measured in a fresh subprocess and appended to a JSONL store, so an
interrupted sweep resumes by re-running the same command.

Examples:
    # Async batching, threads pinned so async is what varies
    ./scripts/benchmark-vec-env.py --suite async --num-threads 24 --repeats 3

    # Sync scaling baseline
    ./scripts/benchmark-vec-env.py --suite scaling --num-threads 24

    # Summarise / plot what has been collected so far
    ./scripts/benchmark-vec-env.py --suite async --summary --plot async.png

Note:
    `num_threads` defaults to `min(batch_size, hardware_concurrency)` inside ALE, so an
    async sweep left unpinned shrinks its worker pool in proportion to `batch_size` and
    measures thread count instead of async. This script therefore pins `--num-threads`
    to the logical core count by default and refuses to run an async sweep with it unset.
"""

from __future__ import annotations

import argparse
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

DEFAULT_GAMES = ("breakout", "pong", "ms_pacman")

BASE_CONFIG: dict[str, Any] = {
    "game": "breakout",
    "num_envs": 128,
    "batch_size": 0,  # 0 -> sync (batch_size == num_envs)
    "num_threads": 0,  # 0 -> ALE default; sweeps should always pin this
    "thread_affinity_offset": -1,
    "autoreset_mode": "NextStep",
    "grayscale": True,
    "frameskip": 4,
    "stack_num": 4,
    "img_size": 84,
    "repeat_action_probability": 0.0,
    "max_num_frames_per_episode": 108_000,
    "api": "step",  # "step" or "send-recv"
}


class ALEBackend:
    """`AtariVectorEnv` wrapped in the benchmark harness' backend protocol."""

    def __init__(self, config: dict[str, Any]):
        """Store the configuration; the environment is built later by `build()`.

        Args:
            config: Configuration keys as in `BASE_CONFIG`.
        """
        self.config = config
        self.name = "ale"
        self.num_envs = int(config["num_envs"])
        requested_batch = int(config.get("batch_size", 0))
        self.batch_size = requested_batch if requested_batch > 0 else self.num_envs
        self.envs: Any = None
        self._send_recv = config.get("api", "step") == "send-recv"
        # Episode-boundary accounting, off by default: it adds a branch and a numpy
        # reduction to the hot loop, which would make SPS incomparable with the other
        # suites. The `reset` suite turns it on because the reset *rate* is its
        # independent variable and must be observed rather than assumed.
        self._count_resets = bool(config.get("count_resets", False))
        self._episode_ends = 0
        self._batches = 0

    def build(self) -> None:
        """Construct the vector environment."""
        from ale_py import AtariVectorEnv

        config = self.config
        self.envs = AtariVectorEnv(
            game=config["game"],
            num_envs=self.num_envs,
            batch_size=int(config.get("batch_size", 0)),
            num_threads=int(config.get("num_threads", 0)),
            thread_affinity_offset=int(config.get("thread_affinity_offset", -1)),
            autoreset_mode=config.get("autoreset_mode", "NextStep"),
            grayscale=bool(config.get("grayscale", True)),
            frameskip=int(config.get("frameskip", 4)),
            stack_num=int(config.get("stack_num", 4)),
            img_height=int(config.get("img_size", 84)),
            img_width=int(config.get("img_size", 84)),
            repeat_action_probability=float(
                config.get("repeat_action_probability", 0.0)
            ),
            max_num_frames_per_episode=int(
                config.get("max_num_frames_per_episode", 108_000)
            ),
        )
        assert self.envs.batch_size == self.batch_size

    def reset(self) -> None:
        """Reset every sub-environment with a fixed seed."""
        self.envs.reset(seed=0)

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> np.ndarray:
        """Return pre-generated discrete action batches of shape `(pool, batch)`."""
        num_actions = self.envs.single_action_space.n
        return rng.integers(
            0, num_actions, size=(pool_size, self.batch_size), dtype=np.int32
        )

    def step_batch(self, actions: np.ndarray) -> np.ndarray:
        """Advance one batch, returning the ids of the environments received."""
        if self._send_recv:
            self.envs.send(actions)
            result = self.envs.recv()
        else:
            result = self.envs.step(actions)
        if self._count_resets:
            _, _, terminations, truncations, _ = result
            self._episode_ends += int(np.count_nonzero(terminations | truncations))
            self._batches += 1
        return result[4]["env_id"]

    @property
    def extra(self) -> dict[str, Any]:
        """Backend-resolved values merged into the run's metadata.

        `episode_ends` spans warm-up as well as the measured window, since the loop
        exposes no hook to zero it at the start of the window. That biases nothing: the
        rate is what is wanted and it is the same in both phases.
        """
        if not self._count_resets or self._batches == 0:
            return {}
        return {
            "episode_ends": self._episode_ends,
            "episode_ends_per_batch": self._episode_ends / self._batches,
            "counted_batches": self._batches,
        }

    def close(self) -> None:
        """Close the vector environment."""
        if self.envs is not None:
            self.envs.close()
            self.envs = None


def run_one(config: dict[str, Any], args: Namespace) -> MeasuredRun:
    """Measure a single configuration in this process."""
    backend = ALEBackend(config)
    return measure(
        backend,
        config,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        frameskip=int(config.get("frameskip", 4)),
        seed=int(config.get("repeat", 0)),
        record_latency=not args.no_latency,
        verify_work=not args.no_verify_work,
    )


# --------------------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------------------


def build_configs(args: Namespace) -> list[dict[str, Any]]:
    """Expand the selected suite into a list of configurations.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Configurations to measure, in run order.

    Raises:
        ValueError: If the suite name is unknown.
    """
    base = dict(BASE_CONFIG)
    base["game"] = args.game
    base["num_threads"] = args.num_threads
    base["api"] = args.api

    suite = args.suite
    configs: list[dict[str, Any]] = []

    if suite == "async":
        num_envs = args.num_envs or [64, 128, 256, 512]
        for n in num_envs:
            for divisor in args.batch_divisors or [1, 2, 3, 4]:
                configs.append({**base, "num_envs": n, "batch_size": n // divisor})
    elif suite == "scaling":
        for n in args.num_envs or [8, 16, 32, 64, 128, 256, 512]:
            configs.append({**base, "num_envs": n, "batch_size": n})
    elif suite == "threads":
        cores = psutil.cpu_count(logical=True) or 8
        physical = psutil.cpu_count(logical=False) or cores // 2
        # Spans the SMT boundary, so per-core efficiency can be compared between
        # `physical` and `logical` thread counts.
        thread_counts = sorted(
            {cores // 4, cores // 2, physical, 3 * cores // 4, cores} - {0}
        )
        for n in args.num_envs or [128]:
            for threads in thread_counts:
                configs.append(
                    {**base, "num_envs": n, "batch_size": n, "num_threads": threads}
                )
    elif suite == "reset":
        # The one knob that changes reset frequency with nothing else moving.
        # Truncation at `max_num_frames_per_episode` forces an autoreset, so caps below
        # Breakout's natural episode (~2,000 frames) raise the rate; 108,000 is the
        # natural rate and acts as the control. The prediction under test is that the
        # straggler tail — `latency.mean - latency.p50` — scales with the reset rate.
        # Each cap is measured synchronously *and* asynchronously, which is what makes
        # this an attribution test rather than a correlation. A reset costs the same CPU
        # in both modes; only sync pays its makespan, because `wait_for_batch` blocks on
        # the slowest env. So if resets drive the tail, sync throughput must fall with
        # reset rate several times faster than async throughput does. If both fall
        # alike, the cost is the reset's own CPU rather than the batch waiting on it.
        caps = args.max_frames or [400, 800, 1600, 3200, 108_000]
        for n in args.num_envs or [64, 128]:
            for cap in caps:
                for batch in (n, n // 4):
                    configs.append(
                        {
                            **base,
                            "num_envs": n,
                            "batch_size": batch,
                            "max_num_frames_per_episode": cap,
                            "count_resets": True,
                        }
                    )
    elif suite == "autoreset":
        for n in args.num_envs or [64, 128, 256, 512]:
            for mode in ("NextStep", "SameStep"):
                configs.append(
                    {**base, "num_envs": n, "batch_size": n, "autoreset_mode": mode}
                )
    elif suite == "obs":
        for n in args.num_envs or [64, 128, 256]:
            for grayscale in (True, False):
                configs.append(
                    {**base, "num_envs": n, "batch_size": n, "grayscale": grayscale}
                )
    elif suite == "games":
        for game in args.games or list(DEFAULT_GAMES):
            for n in args.num_envs or [64, 128, 256, 512]:
                configs.append({**base, "game": game, "num_envs": n, "batch_size": n})
    elif suite == "startup":
        for n in args.num_envs or [8, 16, 32, 64, 128, 256, 512]:
            configs.append({**base, "num_envs": n, "batch_size": n})
    elif suite == "quick":
        for n in args.num_envs or [16, 64]:
            configs.append({**base, "num_envs": n, "batch_size": n})
    else:
        raise ValueError(f"unknown suite: {suite}")

    return configs


def warn_starved_configs(configs: list[dict[str, Any]]) -> None:
    """Warn about async configs that cannot keep the worker pool fed.

    In steady state the number of environments in flight (queued or executing) is
    `num_envs - batch_size`: `recv()` hands `batch_size` of them back to the caller and
    they only re-enter the queue on the next `send()`. So if that margin is below the
    worker count, some workers are starved *by arithmetic* and `dequeue_wait` will stay
    non-zero however well async is implemented. Such points measure the margin, not
    async, and should be excluded from any headline claim.

    Args:
        configs: Configurations about to be measured.
    """
    starved = [
        config
        for config in configs
        if config["batch_size"] < config["num_envs"]
        and config["num_envs"] - config["batch_size"] < config["num_threads"]
    ]
    if not starved:
        return
    print(
        f"\nWARNING: {len(starved)} async config(s) leave fewer envs in flight than "
        f"there are worker threads; residual idle there is arithmetic, not a finding:"
    )
    for config in starved:
        in_flight = config["num_envs"] - config["batch_size"]
        print(
            f"  num_envs={config['num_envs']} batch_size={config['batch_size']}"
            f" -> {in_flight} in flight for {config['num_threads']} threads"
        )
    print()


SUITE_DEFAULTS: dict[str, dict[str, float]] = {
    # startup only needs enough steps to prove the env works; the cost of interest
    # (ctor + first reset) is measured before the window opens.
    "startup": {"warmup_seconds": 0.0, "measure_seconds": 2.0},
    "quick": {"warmup_seconds": 1.0, "measure_seconds": 5.0},
}


def plot(store: ResultStore, path: str) -> None:
    """Plot steps/second against `num_envs`, one line per configuration group."""
    import matplotlib.pyplot as plt

    df = store.to_dataframe()
    if df.empty:
        print("nothing to plot")
        return

    df["ratio"] = (df["cfg.num_envs"] / df["cfg.batch_size"]).round(2)
    label_columns = [
        column
        for column in ("cfg.game", "ratio", "cfg.num_threads", "cfg.autoreset_mode")
        if column in df.columns and df[column].nunique() > 1
    ]
    if not label_columns:
        label_columns = ["cfg.game"]
    df["label"] = df[label_columns].astype(str).agg(" ".join, axis=1)

    fig, (throughput_ax, memory_ax) = plt.subplots(1, 2, figsize=(13, 5))
    for label, group in df.groupby("label"):
        stats = group.groupby("cfg.num_envs")["steps_per_second"].agg(["mean", "std"])
        throughput_ax.errorbar(
            stats.index,
            stats["mean"],
            yerr=stats["std"].fillna(0.0),
            marker="o",
            label=label,
        )
        memory = group.groupby("cfg.num_envs")["peak_rss_mb"].mean()
        memory_ax.plot(memory.index, memory.values, marker="o", label=label)

    throughput_ax.set(xlabel="Number of environments", ylabel="Agent steps / second")
    memory_ax.set(xlabel="Number of environments", ylabel="Peak RSS (MB)")
    for ax in (throughput_ax, memory_ax):
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"wrote {path}")


def parse_args() -> Namespace:
    """Parse the command line."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="quick",
        choices=[
            "async",
            "scaling",
            "threads",
            "reset",
            "autoreset",
            "obs",
            "games",
            "startup",
            "quick",
        ],
    )
    parser.add_argument("--game", default="breakout", help="ROM name (snake_case)")
    parser.add_argument(
        "--games", type=lambda s: s.split(","), help="games for --suite games"
    )
    parser.add_argument(
        "--num-envs",
        type=lambda s: list(iter_int_list(s)),
        help="override the suite's env counts",
    )
    parser.add_argument(
        "--max-frames",
        type=lambda s: list(iter_int_list(s)),
        help="reset: max_num_frames_per_episode values to sweep",
    )
    parser.add_argument(
        "--batch-divisors",
        type=lambda s: list(iter_int_list(s)),
        help="async: batch_size = num_envs // divisor",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=psutil.cpu_count(logical=True),
        help="worker threads, pinned across the sweep (0 = ALE default, discouraged)",
    )
    parser.add_argument("--api", default="step", choices=["step", "send-recv"])
    parser.add_argument("--warmup-seconds", type=float)
    parser.add_argument("--measure-seconds", type=float)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--out", help="JSONL results path (default: benchmark-results/<suite>.jsonl)"
    )
    parser.add_argument(
        "--no-isolate",
        action="store_true",
        help="run in-process instead of a fresh subprocess per config",
    )
    parser.add_argument(
        "--no-latency", action="store_true", help="skip per-iteration timing"
    )
    parser.add_argument(
        "--no-verify-work", action="store_true", help="skip the per-env recv histogram"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the configurations the suite would run and exit",
    )
    parser.add_argument(
        "--summary", action="store_true", help="summarise the existing store and exit"
    )
    parser.add_argument("--plot", help="write a plot of the store to this path")
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    defaults = SUITE_DEFAULTS.get(args.suite, {})
    if args.warmup_seconds is None:
        args.warmup_seconds = defaults.get("warmup_seconds", 8.0)
    if args.measure_seconds is None:
        args.measure_seconds = defaults.get("measure_seconds", 60.0)
    return args


def main() -> None:
    """Entry point: run one configuration, or drive a whole suite."""
    args = parse_args()

    if args.run_one:
        emit_result(run_one(json.loads(args.run_one), args))
        return

    out = Path(args.out) if args.out else DEFAULT_RESULTS_DIR / f"{args.suite}.jsonl"
    store = ResultStore(out)

    if args.summary or args.plot:
        summarise(
            store,
            [
                "game",
                "num_envs",
                "batch_size",
                "num_threads",
                "autoreset_mode",
                "grayscale",
            ],
        )
        if args.plot:
            plot(store, args.plot)
        return

    configs = build_configs(args)
    if args.suite == "async" and args.num_threads <= 0:
        raise SystemExit(
            "refusing to run an async sweep with --num-threads 0: ALE would size the "
            "worker pool from batch_size, so the sweep would measure thread count "
            "rather than async batching. Pass --num-threads <logical cores>."
        )

    print(f"suite={args.suite} configs={len(configs)} repeats={args.repeats}")
    print(
        f"warmup={args.warmup_seconds}s measure={args.measure_seconds}s threads={args.num_threads}"
    )
    estimate = (
        len(configs) * args.repeats * (args.warmup_seconds + args.measure_seconds + 10)
    )
    print(f"rough wall-clock estimate: {estimate / 60:.0f} min (excludes construction)")
    warn_starved_configs(configs)

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
    )

    print()
    summarise(
        store,
        [
            "game",
            "num_envs",
            "batch_size",
            "num_threads",
            "autoreset_mode",
            "grayscale",
        ],
    )


if __name__ == "__main__":
    main()
