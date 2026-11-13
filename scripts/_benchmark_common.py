"""Shared measurement harness for the vector-environment benchmark scripts.

Used by `benchmark-vec-env.py` (ALE-only sweeps) and
`benchmark-vec-env-comparison.py` (ALE vs EnvPool / PufferLib / Gymnasium).

It keeps both scripts consistent on the things that are easy to get wrong when
benchmarking a vector environment:

* **Step accounting.** One `recv()` yields `batch_size` transitions, *not* `num_envs`.
  Counting `num_envs` in async mode (`batch_size < num_envs`) inflates throughput by
  exactly `num_envs / batch_size`, which would manufacture the async speed-up the
  measurement is meant to test.
* **Thread pinning.** ALE's `num_threads` defaults to `min(batch_size, hardware_concurrency)`,
  so an async sweep that leaves it unset measures thread count instead of async.
  `MeasuredRun` records the value actually used so a sweep can never silently vary it.
* **Isolation.** Each configuration runs in a fresh subprocess so worker pools, page
  cache and allocator state never carry over, and peak RSS is attributable.
* **Verified work.** Optional per-env `recv` histograms confirm two APIs did the same
  amount of emulation before their throughputs are compared.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import psutil

RESULT_PREFIX = "@@BENCHMARK_RESULT@@ "
"""Marker a child process prints its JSON result behind, so stdout can stay chatty."""

DEFAULT_ACTION_POOL = 512
"""Number of pre-generated action batches, cycled through during measurement.

Pre-generating keeps RNG cost out of the timed loop; cycling a pool rather than
using one fixed batch keeps the action stream non-degenerate.
"""


# --------------------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------------------


class VectorBackend(Protocol):
    """Minimal interface a benchmarked vector environment must provide.

    Implementations are constructed cheaply; all expensive setup belongs in `build()`
    so that construction cost is measured rather than hidden.

    An implementation may also expose an `extra` dict of values it resolved at build
    time; those are merged into the run's metadata.
    """

    name: str
    num_envs: int
    batch_size: int

    def build(self) -> None:
        """Construct the underlying vector environment (timed as constructor cost)."""

    def reset(self) -> None:
        """Perform the initial reset, which is timed separately from construction."""

    def action_pool(self, rng: np.random.Generator, pool_size: int) -> Any:
        """Return `pool_size` pre-generated action batches, indexable by integer."""

    def step_batch(self, actions: Any) -> np.ndarray | None:
        """Advance one batch and return the env ids received, or None if untracked."""

    def close(self) -> None:
        """Release the environment and its workers."""


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------


@dataclasses.dataclass
class MeasuredRun:
    """Outcome of a single timed measurement window."""

    config: dict[str, Any]
    steps: int
    iterations: int
    elapsed: float
    steps_per_second: float
    frames_per_second: float
    ctor_seconds: float
    first_reset_seconds: float
    peak_rss_mb: float
    rss_delta_mb: float
    latency_ms: dict[str, float]
    work_check: dict[str, Any] | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of the run."""
        return dataclasses.asdict(self)


def config_key(config: dict[str, Any]) -> str:
    """Return a stable key for a configuration, used for resume/dedup."""
    return json.dumps(config, sort_keys=True, default=str)


DEFAULT_RESULTS_DIR = Path("benchmark-results")
"""Where results are written when no `--out` is given, relative to the working directory."""


class ResultStore:
    """Append-only JSONL store of `MeasuredRun` records with resume support."""

    def __init__(self, path: str | os.PathLike[str]):
        """Open (or create) the store at `path` and index any existing records.

        Args:
            path: Destination `.jsonl` file. Parent directories are created.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        self._keys = {config_key(r["config"]) for r in self.records}

    def __contains__(self, config: dict[str, Any]) -> bool:
        """Return whether a run for `config` is already stored."""
        return config_key(config) in self._keys

    def append(self, record: dict[str, Any]) -> None:
        """Append a record and flush it to disk immediately."""
        self.records.append(record)
        self._keys.add(config_key(record["config"]))
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def to_dataframe(self):
        """Return a flat pandas DataFrame of all stored records."""
        import pandas as pd

        rows = []
        for record in self.records:
            row = {f"cfg.{k}": v for k, v in record["config"].items()}
            row.update(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("config", "latency_ms", "work_check", "metadata")
                }
            )
            row.update({f"latency.{k}": v for k, v in record["latency_ms"].items()})
            rows.append(row)
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Resource sampling
# --------------------------------------------------------------------------------------


class RSSMonitor:
    """Background sampler for peak resident memory of this process and its children.

    A sampler is used rather than `getrusage` alone because process-based backends
    (PufferLib, Gymnasium's `AsyncVectorEnv`) hold most of their memory in children,
    and because peak RSS is the figure the comparison report quotes.
    """

    def __init__(self, interval: float = 0.1):
        """Create a monitor sampling every `interval` seconds."""
        self.interval = interval
        self.peak_mb = 0.0
        self.baseline_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process()

    def _sample(self) -> float:
        total = 0
        try:
            total += self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total / 1e6

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self.peak_mb = max(self.peak_mb, self._sample())

    def start(self) -> None:
        """Record the baseline and begin sampling."""
        self.baseline_mb = self._sample()
        self.peak_mb = self.baseline_mb
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop sampling, taking one final sample."""
        self.peak_mb = max(self.peak_mb, self._sample())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def machine_metadata() -> dict[str, Any]:
    """Return machine/software provenance recorded alongside every run."""
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cores": psutil.cpu_count(logical=True),
        "physical_cores": psutil.cpu_count(logical=False),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        meta["git_commit"] = None
    try:
        import ale_py

        meta["ale_py"] = ale_py.__version__
    except ImportError:
        meta["ale_py"] = None
    return meta


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


def _percentiles(samples: np.ndarray) -> dict[str, float]:
    if samples.size == 0:
        return {}
    p50, p90, p99, p999 = np.percentile(samples, [50, 90, 99, 99.9])
    return {
        "mean": float(samples.mean() * 1e3),
        "p50": float(p50 * 1e3),
        "p90": float(p90 * 1e3),
        "p99": float(p99 * 1e3),
        "p999": float(p999 * 1e3),
        "max": float(samples.max() * 1e3),
    }


def measure(
    backend: VectorBackend,
    config: dict[str, Any],
    *,
    warmup_seconds: float = 8.0,
    measure_seconds: float = 60.0,
    min_iterations: int = 10,
    frameskip: int = 4,
    seed: int = 0,
    action_pool: int = DEFAULT_ACTION_POOL,
    record_latency: bool = True,
    verify_work: bool = True,
) -> MeasuredRun:
    """Time one configuration end to end and return its measurement.

    The timed loop contains only `step_batch`, an index increment and (optionally) two
    `perf_counter` calls, so the measurement is dominated by the backend rather than the
    harness. Actions are pre-generated before the clock starts.

    Args:
        backend: Backend to build, warm up and measure.
        config: Configuration description recorded verbatim with the result.
        warmup_seconds: Wall-clock warm-up before the measurement window.
        measure_seconds: Length of the measurement window.
        min_iterations: Lower bound on iterations, in case the window is very short.
        frameskip: Emulator frames per agent step, used to derive frames/second.
        seed: Seed for the pre-generated action stream.
        action_pool: Number of distinct action batches to cycle through.
        record_latency: Time each iteration to report per-`recv` latency percentiles.
        verify_work: Accumulate a per-env `recv` histogram to confirm work done.

    Returns:
        The completed `MeasuredRun`.
    """
    rng = np.random.default_rng(seed)
    monitor = RSSMonitor()
    monitor.start()

    t0 = time.perf_counter()
    backend.build()
    ctor_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    backend.reset()
    first_reset_seconds = time.perf_counter() - t0

    pool = backend.action_pool(rng, action_pool)
    pool_size = len(pool)

    env_counts = np.zeros(backend.num_envs, dtype=np.int64) if verify_work else None
    tracked_ids = verify_work

    index = 0
    deadline = time.perf_counter() + warmup_seconds
    while time.perf_counter() < deadline:
        backend.step_batch(pool[index % pool_size])
        index += 1

    latencies: list[float] = []
    steps = 0
    iterations = 0
    index = 0

    start = time.perf_counter()
    deadline = start + measure_seconds
    while True:
        actions = pool[index % pool_size]
        if record_latency:
            before = time.perf_counter()
            env_ids = backend.step_batch(actions)
            now = time.perf_counter()
            latencies.append(now - before)
        else:
            env_ids = backend.step_batch(actions)
            now = time.perf_counter()

        if tracked_ids:
            if env_ids is None:
                tracked_ids = False
            else:
                env_counts += np.bincount(
                    np.asarray(env_ids, dtype=np.int64), minlength=backend.num_envs
                )

        steps += backend.batch_size
        iterations += 1
        index += 1
        if now >= deadline and iterations >= min_iterations:
            break

    elapsed = time.perf_counter() - start
    monitor.stop()
    backend.close()

    work_check: dict[str, Any] | None = None
    if env_counts is not None and tracked_ids:
        work_check = {
            "recvs_per_env_min": int(env_counts.min()),
            "recvs_per_env_max": int(env_counts.max()),
            "recvs_per_env_mean": float(env_counts.mean()),
            "total_env_steps": int(env_counts.sum()),
            "matches_step_count": bool(int(env_counts.sum()) == steps),
        }

    return MeasuredRun(
        config=config,
        steps=steps,
        iterations=iterations,
        elapsed=elapsed,
        steps_per_second=steps / elapsed,
        frames_per_second=steps * frameskip / elapsed,
        ctor_seconds=ctor_seconds,
        first_reset_seconds=first_reset_seconds,
        peak_rss_mb=monitor.peak_mb,
        rss_delta_mb=monitor.peak_mb - monitor.baseline_mb,
        latency_ms=_percentiles(np.asarray(latencies)) if record_latency else {},
        work_check=work_check,
        # `backend.extra` carries anything the backend resolved at build time (e.g. the
        # worker count it actually settled on). It goes in metadata rather than config
        # because config is the resume key and must stay exactly what was requested.
        metadata={**machine_metadata(), **getattr(backend, "extra", {})},
    )


# --------------------------------------------------------------------------------------
# Subprocess isolation
# --------------------------------------------------------------------------------------


def emit_result(run: MeasuredRun) -> None:
    """Print a run so an isolating parent process can parse it back out."""
    sys.stdout.write(RESULT_PREFIX + json.dumps(run.to_dict(), default=str) + "\n")
    sys.stdout.flush()


def run_isolated(
    script: str | os.PathLike[str],
    config: dict[str, Any],
    *,
    extra_args: Sequence[str] = (),
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Run one configuration in a fresh subprocess and return its result record.

    Args:
        script: Benchmark script to re-invoke (it must support `--run-one`).
        config: Configuration passed through as JSON.
        extra_args: Additional CLI arguments for the child.
        timeout: Optional wall-clock limit for the child.
        env: Optional environment overrides for the child.

    Returns:
        The parsed result record.

    Raises:
        RuntimeError: If the child fails or prints no result line.
    """
    cmd = [
        sys.executable,
        str(script),
        "--run-one",
        json.dumps(config, default=str),
        *extra_args,
    ]
    child_env = {**os.environ, **(env or {})}
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=child_env
    )
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            return json.loads(line[len(RESULT_PREFIX) :])
    raise RuntimeError(
        f"no result from child (exit {proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-4000:]}"
    )


def sweep(
    script: str | os.PathLike[str],
    configs: Iterable[dict[str, Any]],
    store: ResultStore,
    *,
    repeats: int = 1,
    isolate: bool = True,
    extra_args: Sequence[str] = (),
    inline_runner: Callable[[dict[str, Any]], MeasuredRun] | None = None,
    timeout: float | None = None,
) -> None:
    """Run every configuration (optionally repeated) and store the results.

    Configurations already present in `store` are skipped, so an interrupted sweep can
    be resumed by re-running the same command.

    Args:
        script: Script to re-invoke for isolated runs.
        configs: Configurations to measure.
        store: Destination store; also provides resume state.
        repeats: Independent repeats per configuration (recorded as `repeat`).
        isolate: Run each configuration in a fresh subprocess.
        extra_args: Extra CLI arguments for isolated children.
        inline_runner: Runner used when `isolate` is False.
        timeout: Optional per-run wall-clock limit.

    Raises:
        ValueError: If `isolate` is False and no `inline_runner` was given.
    """
    if not isolate and inline_runner is None:
        raise ValueError("inline_runner is required when isolate=False")

    pending = []
    for config in configs:
        for repeat in range(repeats):
            candidate = {**config, "repeat": repeat}
            if candidate not in store:
                pending.append(candidate)

    print(f"{len(pending)} run(s) to go (store: {store.path})")
    for i, candidate in enumerate(pending, start=1):
        label = ", ".join(f"{k}={v}" for k, v in candidate.items())
        print(f"\n[{i}/{len(pending)}] {label}", flush=True)
        started = time.perf_counter()
        if isolate:
            record = run_isolated(
                script, candidate, extra_args=extra_args, timeout=timeout
            )
        else:
            assert inline_runner is not None
            record = inline_runner(candidate).to_dict()
        store.append(record)
        latency = record.get("latency_ms") or {}
        # mean/p50 is the straggler-tail diagnostic: in a synchronous batch the mean is
        # dragged above the median by whichever env hit a reset, so a ratio near 1.0 is
        # the signature of a batch that no longer waits on outliers.
        skew = (
            f"  |  batch mean/p50 {latency['mean'] / latency['p50']:.2f}"
            if latency.get("p50")
            else ""
        )
        print(
            f"    {record['steps_per_second']:>10,.0f} steps/s"
            f"  |  {record['frames_per_second']:>10,.0f} frames/s"
            f"  |  ctor {record['ctor_seconds']:.2f}s"
            f"  |  peak RSS {record['peak_rss_mb']:.0f} MB"
            f"{skew}"
            f"  |  {time.perf_counter() - started:.0f}s wall",
            flush=True,
        )


def summarise(store: ResultStore, group_keys: Sequence[str]) -> None:
    """Print a mean ± std (CV) summary of the store grouped by `group_keys`."""
    df = store.to_dataframe()
    if df.empty:
        print("no results yet")
        return
    columns = [f"cfg.{k}" for k in group_keys if f"cfg.{k}" in df.columns]
    grouped = df.groupby(columns)["steps_per_second"].agg(["mean", "std", "count"])
    grouped["cv%"] = 100 * grouped["std"] / grouped["mean"]
    print(grouped.to_string(float_format=lambda v: f"{v:,.1f}"))


def iter_int_list(text: str) -> Iterator[int]:
    """Yield the integers in a comma-separated CLI argument."""
    for part in text.split(","):
        part = part.strip()
        if part:
            yield int(part)
