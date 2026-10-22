#!/usr/bin/env python3
"""Throughput benchmark for the XLA / JAX FFI path of `AtariVectorEnv`.

Four ways of driving the same vectorizer are compared at an identical configuration, so
the differences are attributable to the calling path rather than to the emulator:

| mode | what it measures |
|---|---|
| `numpy` | `env.step()`, the baseline every other report uses |
| `xla_eager` | `xla_step` called directly, one FFI dispatch per batch, no compilation |
| `xla_jit` | `jax.jit(xla_step)`, one compiled dispatch per batch |
| `xla_scan` | `jax.lax.scan` of `--scan-length` steps inside one compiled call |

`xla_scan` is the idiomatic JAX rollout and the only mode that amortises dispatch across
steps, so the `xla_jit` → `xla_scan` gap is the cost of re-entering XLA per step.

On a CUDA build the default JAX platform is the GPU, so the observations are copied to
the device every batch and the numbers are not comparable with the CPU backend's. Pass
`--platform cpu` for a like-for-like comparison against the numpy path, or
`--platform cuda` to measure the GPU path explicitly. The platform is recorded with
each result.

Usage:
    ./scripts/benchmark-vec-env-xla.py --num-envs 64,128,256 --repeats 2
    ./scripts/benchmark-vec-env-xla.py --platform cpu --modes numpy,xla_jit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT = Path(__file__).resolve()
DEFAULT_OUT = Path("benchmark-results") / "xla.jsonl"
MODES = ("numpy", "xla_eager", "xla_jit", "xla_scan")
POOL = 64  # distinct pre-generated action batches, cycled during the measured loop


def run_one(config: dict[str, Any]) -> dict[str, Any]:
    """Measure one (mode, num_envs, batch_size) point in this process."""
    import numpy as np
    from ale_py import AtariVectorEnv

    n, batch, mode = config["num_envs"], config["batch_size"], config["mode"]
    warmup_s, measure_s = config["warmup_seconds"], config["measure_seconds"]
    scan_len = config["scan_length"]

    t0 = time.perf_counter()
    env = AtariVectorEnv(
        config["game"],
        n,
        batch_size=0 if batch == n else batch,
        num_threads=config["num_threads"],
    )
    ctor_seconds = time.perf_counter() - t0
    num_actions = env.single_action_space.n

    # A pre-generated pool of random action batches, cycled by index. All-NOOP actions
    # would be unrepresentative: Breakout never launches the ball without FIRE, so the
    # episode never ends and the reset path — a sixth of synchronous wall — never runs.
    rng = np.random.default_rng(0)
    pool_np = rng.integers(0, num_actions, size=(POOL, batch), dtype=np.int32)

    if mode == "numpy":
        env.reset(seed=0)
        index = 0

        def one_batch() -> int:
            nonlocal index
            env.step(pool_np[index % POOL])
            index += 1
            return batch

        steps_per_call = batch
    else:
        import jax
        import jax.numpy as jnp

        handle, xla_reset, xla_step = env.xla()
        # No seed: `xla_reset` expects one seed per env being reset, and seeding is not
        # what is being measured.
        handle, _ = xla_reset(handle)
        # Device-resident so the host->device copy is not timed per step; the numpy path
        # likewise pre-generates its actions.
        pool = jnp.asarray(pool_np)
        index = 0

        if mode == "xla_eager":
            state = handle

            def one_batch() -> int:
                nonlocal state, index
                state, out = xla_step(state, pool[index % POOL])
                index += 1
                out[1].block_until_ready()
                return batch

            steps_per_call = batch
        elif mode == "xla_jit":
            jitted = jax.jit(xla_step)
            state = handle

            def one_batch() -> int:
                nonlocal state, index
                state, out = jitted(state, pool[index % POOL])
                index += 1
                out[1].block_until_ready()
                return batch

            steps_per_call = batch
        else:  # xla_scan
            scan_actions = jnp.asarray(
                rng.integers(0, num_actions, size=(scan_len, batch), dtype=np.int32)
            )

            def body(carry, acts):
                new_handle, out = xla_step(carry, acts)
                # Carry only the handle and return rewards; keeping observations would
                # make the scan allocate `scan_length` full observation stacks.
                return new_handle, out[1]

            scanned = jax.jit(lambda h, xs: jax.lax.scan(body, h, xs))
            state = handle

            def one_batch() -> int:
                nonlocal state
                state, rewards = scanned(state, scan_actions)
                rewards.block_until_ready()
                return batch * scan_len

            steps_per_call = batch * scan_len

    deadline = time.perf_counter() + warmup_s
    while time.perf_counter() < deadline:
        one_batch()

    latencies: list[float] = []
    steps = 0
    start = time.perf_counter()
    deadline = start + measure_s
    while True:
        before = time.perf_counter()
        steps += one_batch()
        now = time.perf_counter()
        latencies.append(now - before)
        if now >= deadline:
            break
    elapsed = time.perf_counter() - start
    env.close()

    latencies.sort()
    return {
        **config,
        "steps": steps,
        "iterations": len(latencies),
        "steps_per_call": steps_per_call,
        "elapsed": elapsed,
        "steps_per_second": steps / elapsed,
        "frames_per_second": steps * 4 / elapsed,
        "ctor_seconds": ctor_seconds,
        "call_ms_p50": latencies[len(latencies) // 2] * 1e3,
        "call_ms_p99": latencies[int(len(latencies) * 0.99)] * 1e3,
    }


def main() -> None:
    """Drive the sweep, one configuration per subprocess."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", default="breakout")
    parser.add_argument(
        "--num-envs",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[64, 128, 256],
    )
    parser.add_argument(
        "--batch-divisors",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[1],
        help="batch_size = num_envs // divisor; 1 is synchronous",
    )
    parser.add_argument(
        "--modes", type=lambda s: s.split(","), default=list(MODES), choices=None
    )
    parser.add_argument("--num-threads", type=int, default=os.cpu_count())
    parser.add_argument("--scan-length", type=int, default=16)
    parser.add_argument("--warmup-seconds", type=float, default=8.0)
    parser.add_argument("--measure-seconds", type=float, default=30.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--platform", default=None, help="force JAX_PLATFORMS, e.g. cpu"
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--run-one", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.platform:
        os.environ["JAX_PLATFORMS"] = args.platform

    if args.run_one:
        print("RESULT " + json.dumps(run_one(json.loads(args.run_one))), flush=True)
        return

    import subprocess

    configs = [
        {
            "game": args.game,
            "num_envs": n,
            "batch_size": n // d,
            "mode": mode,
            "num_threads": args.num_threads,
            "scan_length": args.scan_length,
            "warmup_seconds": args.warmup_seconds,
            "measure_seconds": args.measure_seconds,
            "platform": args.platform or "default",
            "repeat": r,
        }
        for n in args.num_envs
        for d in args.batch_divisors
        for mode in args.modes
        for r in range(args.repeats)
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(configs)} run(s) -> {out}")

    with out.open("a") as fh:
        for i, config in enumerate(configs, 1):
            print(
                f"[{i}/{len(configs)}] mode={config['mode']} n={config['num_envs']} "
                f"batch={config['batch_size']} repeat={config['repeat']}",
                flush=True,
            )
            proc = subprocess.run(
                [sys.executable, str(SCRIPT), "--run-one", json.dumps(config)]
                + (["--platform", args.platform] if args.platform else []),
                capture_output=True,
                text=True,
                timeout=1800,
            )
            line = next(
                (x for x in proc.stdout.splitlines() if x.startswith("RESULT ")), None
            )
            if line is None:
                print(f"  FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
                continue
            record = json.loads(line[len("RESULT ") :])
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            print(
                f"  {record['steps_per_second']:>10,.0f} steps/s  "
                f"p50 {record['call_ms_p50']:.2f} ms  ctor {record['ctor_seconds']:.1f}s"
            )


if __name__ == "__main__":
    main()
