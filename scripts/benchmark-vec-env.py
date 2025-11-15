#!/usr/bin/env python3
"""Comprehensive benchmarking script for AtariVectorEnv performance testing.

Allows testing of the impact of `num_envs`, `batch_size`, `autoreset_mode`, and `num_threads` on steps per second and resource usage.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import os
import time
from argparse import ArgumentParser
from collections.abc import Callable
from typing import Any

import gymnasium.vector
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from ale_py import AtariVectorEnv


@dataclasses.dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    num_envs: int

    env_params: dict[str, Any]
    env_params_meaning: str

    mean_steps_per_second: Any
    std_steps_per_second: Any
    mean_cpu_usage: Any
    mean_memory_usage: Any


class BenchmarkAtariVectorEnv:
    """Benchmark class for AtariVectorEnv."""

    def __init__(
        self,
        num_envs_configs: list[int],
        env_params_configs: dict[str, list[tuple[str, Any | Callable[[int], Any]]]],
        results_filename: str,
        game: str = "breakout",
        warmup_steps: int = 100,
        measurement_time: float = 10.0,
        num_runs: int = 3,
    ):
        """Initialize the benchmark suite.

        Args:
            num_envs_configs: List of number of environments for testing
            env_params_configs: Dictionary of environment parameter with list of testing values
            results_filename: Filename for the results to be saved to (csv)
            game: Atari game to test
            warmup_steps: Steps to run before measurement (for JIT warmup)
            measurement_time: Time in seconds to measure for each run
            num_runs: Number of independent runs for statistical significance
        """
        self.num_envs_configs = num_envs_configs
        self.env_params_configs = env_params_configs

        self.results_filename = results_filename
        if os.path.exists(self.results_filename):
            self.results_df = pd.read_csv(self.results_filename, index_col=False)
        else:
            self.results_df = pd.DataFrame(
                columns=[f.name for f in dataclasses.fields(BenchmarkResult)]
            )

        self.game = game
        self.warmup_steps = warmup_steps
        self.measurement_time = measurement_time
        self.num_runs = num_runs

    @staticmethod
    def measure_system_resources() -> tuple[float, float]:
        """Measure current CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        return cpu_percent, memory_percent

    def single_benchmark_run(
        self, num_envs: int, env_params: dict[str, Any]
    ) -> tuple[float, float, float]:
        """Run a single benchmark and return steps per second."""
        _, mem_initial = self.measure_system_resources()

        envs = AtariVectorEnv(self.game, num_envs, **env_params)

        # Warmup phase
        envs.reset()
        for _ in range(self.warmup_steps):
            actions = envs.action_space.sample()
            envs.step(actions)

        # Measurement phase - run for fixed time duration
        step_count = 0
        start_time = time.perf_counter()
        cpu_start, mem_start = self.measure_system_resources()

        while (time.perf_counter() - start_time) < self.measurement_time:
            actions = envs.action_space.sample()
            envs.step(actions)
            step_count += num_envs

        end_time = time.perf_counter()
        cpu_end, mem_end = self.measure_system_resources()

        elapsed_time = end_time - start_time
        steps_per_second = step_count / elapsed_time

        # Average resource usage during benchmark
        avg_cpu = (cpu_start + cpu_end) / 2
        avg_memory = (mem_start + mem_end) / 2 - mem_initial

        envs.close()
        gc.collect()

        return steps_per_second, avg_cpu, avg_memory

    def benchmark_configuration(
        self, num_envs: int, env_params: dict[str, Any], env_params_meaning: str
    ) -> BenchmarkResult:
        """Run multiple benchmark runs for a configuration and return aggregated results."""
        print(f"Benchmarking configuration: num-envs={num_envs}, {env_params}")
        run_results, cpu_usages, memory_usages = [], [], []
        for run in range(self.num_runs):
            print(f"\tRun {run + 1}/{self.num_runs}")
            sps, cpu, memory = self.single_benchmark_run(num_envs, env_params)
            run_results.append(sps)
            cpu_usages.append(cpu)
            memory_usages.append(memory)

            # Brief pause between runs
            time.sleep(1)

        mean_sps = np.mean(run_results)
        std_sps = np.std(run_results)
        mean_cpu = np.mean(cpu_usages)
        mean_memory = np.mean(memory_usages)

        result = BenchmarkResult(
            num_envs=num_envs,
            env_params=env_params,
            env_params_meaning=env_params_meaning,
            mean_steps_per_second=mean_sps,
            std_steps_per_second=std_sps,
            mean_cpu_usage=mean_cpu,
            mean_memory_usage=mean_memory,
        )

        print(f"\tResult: {mean_sps:.1f} ± {std_sps:.1f} steps/sec")
        return result

    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        configurations = [
            (
                num_envs,
                param_meaning,
                {param: value(num_envs) if callable(value) else value},
            )
            for param, values in self.env_params_configs.items()
            for param_meaning, value in values
            for num_envs in self.num_envs_configs
        ]
        completed_configs = [
            (
                row["num_envs"],
                row["env_params_meaning"],
                json.loads(row["env_params"].replace("'", '"')),
            )
            for _, row in self.results_df.iterrows()
        ]
        remaining_configs = [
            config for config in configurations if config not in completed_configs
        ]

        total_configs = len(configurations)
        completed_count = len(completed_configs)
        remaining_count = len(remaining_configs)

        print(f"Total configurations: {total_configs}")
        print(f"Already completed: {completed_count}")
        print(f"Remaining to run: {remaining_count}")

        for i, (num_envs, param_meaning, env_config) in enumerate(remaining_configs):
            print(
                f"\nProgress: {i + 1}/{remaining_count} (Overall: {completed_count + i + 1}/{total_configs})"
            )
            benchmark_result = self.benchmark_configuration(
                num_envs, env_config, param_meaning
            )
            self.results_df = pd.concat(
                (
                    self.results_df,
                    pd.DataFrame(
                        data={
                            k: [v]
                            for k, v in dataclasses.asdict(benchmark_result).items()
                        }
                    ),
                )
            )

            self.results_df.to_csv(self.results_filename, index=False)

    def plot_steps_per_second(
        self,
        save_path: str | None = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """Plot steps per second with error bars from BenchmarkResult data.

        Args:
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        for param_meaning, group in self.results_df.groupby("env_params_meaning"):
            ax.errorbar(
                group["num_envs"],
                group["mean_steps_per_second"],
                yerr=group["std_steps_per_second"],
                label=param_meaning,
            )

        ax.set_xlabel("Number of environments")
        ax.set_ylabel("Steps per Second")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_resource_usage(
        self,
        save_path: str | None = None,
        title: str | None = "Resource Usage",
        figsize: tuple[int, int] = (12, 8),
        colors: list[str] | None = None,
    ) -> plt.Figure:
        """Plot resource usage metrics from BenchmarkResult data.

        Args:
            save_path: Path to save the plot (optional)
            title: Plot title (optional)
            figsize: Figure size tuple
            colors: List of colors for different metrics (optional)

        Returns:
            matplotlib Figure object
        """
        fig, axs = plt.subplots(1, 2, figsize=figsize, sharex=True)

        for param_meaning, group in self.results_df.groupby("env_params_meaning"):
            axs[0].plot(
                group["num_envs"],
                group["mean_cpu_usage"],
                label=param_meaning,
            )
            axs[1].plot(
                group["num_envs"], group["mean_memory_usage"], label=param_meaning
            )

        axs[0].set_title("CPU Usage")
        axs[1].set_title("Memory Usage")
        axs[0].set_xlabel("Number of environments")
        axs[1].set_xlabel("Number of environments")
        axs[0].set_ylabel("CPU Usage (%)")
        axs[1].set_ylabel("Memory Usage (MB)")
        axs[0].grid(True, alpha=0.3)
        axs[1].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--complete", default=False)

    args = parser.parse_args()
    if args.complete:
        # Complete parameter testing
        full_testing_num_envs = [8, 16, 32, 64, 128, 256]
        full_testing_env_params = {
            "batch_size": [
                ("Async /4", lambda n: n // 4),
                ("Async /3", lambda n: n // 3),
                ("Async /2", lambda n: n // 2),
                ("Sync", lambda n: n),
            ],
            "autoreset_mode": [
                ("Next-step", gymnasium.vector.AutoresetMode.NEXT_STEP),
                ("Same-step", gymnasium.vector.AutoresetMode.SAME_STEP),
            ],
        }
        benchmark = BenchmarkAtariVectorEnv(
            full_testing_num_envs,
            full_testing_env_params,
            "full-testing.csv",
            game="breakout",
            warmup_steps=1000,
            measurement_time=10.0,
            num_runs=3,
        )
        benchmark.run_all_benchmarks()

    else:
        limited_testing_num_envs = [16, 64, 128]
        limited_testing_env_params = {"batch_size": [("Sync", lambda n: n)]}
        benchmark = BenchmarkAtariVectorEnv(
            limited_testing_num_envs,
            limited_testing_env_params,
            "limited-results.csv",
            game="breakout",
            warmup_steps=1000,
            measurement_time=10.0,
            num_runs=3,
        )
        benchmark.run_all_benchmarks()

        # Create visualizations
        benchmark.plot_steps_per_second("benchmark-limited-sps.png")
        benchmark.plot_resource_usage("benchmark-limited-resource-usage.png")
