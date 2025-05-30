#!/usr/bin/env python3
"""
Comprehensive benchmarking script for AtariVectorEnv performance testing.
Tests the impact of num_envs, batch_size, autoreset_mode, and num_threads on steps per second.
"""
from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from ale_py import AtariVectorEnv


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    num_envs: int
    env_params: dict[str, Any]

    mean_steps_per_second: Any
    std_steps_per_second: Any
    mean_cpu_usage: Any
    mean_memory_usage: Any


class BenchmarkAtariVectorEnv:
    def __init__(
        self,
        num_envs_configs: list[int],
        env_params_configs: list[tuple[str, list[Any | Callable[[int], Any]]]],
        results_filename: str = "benchmark_results.csv",
        game: str = "breakout",
        warmup_steps: int = 100,
        measurement_time: float = 10.0,
        num_runs: int = 3,
    ):
        """Initialize the benchmark suite.

        Args:
            game: Atari game to test
            warmup_steps: Steps to run before measurement (for JIT warmup)
            measurement_time: Time in seconds to measure for each run
            num_runs: Number of independent runs for statistical significance
        """
        self.num_envs_configs = num_envs_configs
        self.env_params_configs = env_params_configs
        self.results_filename = results_filename

        self.game = game
        self.warmup_steps = warmup_steps
        self.measurement_time = measurement_time
        self.num_runs = num_runs

        if os.path.exists(self.results_filename):
            self.results_df = pd.read_csv(self.results_filename)
        else:
            self.results_df = pd.DataFrame(columns=["num_envs", "param", "value", "mean_sps", "std_sps", "cpu_usage", "memory_usage"])

    @staticmethod
    def measure_system_resources() -> tuple[float, float]:
        """Measure current CPU and memory usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        return cpu_percent, memory_percent

    def single_benchmark_run(
        self, num_envs: int, params: dict[str, Any], rom_id: str = "Breakout"
    ) -> tuple[float, float, float]:
        """Run a single benchmark and return steps per second."""
        envs = AtariVectorEnv(rom_id, num_envs, **params)

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
        avg_memory = (mem_start + mem_end) / 2

        envs.close()
        gc.collect()

        return steps_per_second, avg_cpu, avg_memory

    def benchmark_configuration(self, num_envs: int, env_params: dict[str, Any]) -> BenchmarkResult:
        """Run multiple benchmark runs for a configuration and return aggregated results."""
        print(f"Benchmarking configuration: {env_params}")

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
            mean_steps_per_second=mean_sps,
            std_steps_per_second=std_sps,
            mean_cpu_usage=mean_cpu,
            mean_memory_usage=mean_memory,
        )

        print(f"\tResult: {mean_sps:.1f} ± {std_sps:.1f} steps/sec")
        return result

    def run_all_benchmarks(self):
        """Run all benchmark configurations."""
        configurations = []
        completed_configs = []
        remaining_configs = [config for config in configurations if config not in completed_configs]

        total_configs = len(configurations)
        completed_count = len(completed_configs)
        remaining_count = len(remaining_configs)

        print(f"Total configurations: {total_configs}")
        print(f"Already completed: {completed_count}")
        print(f"Remaining to run: {remaining_count}")

        for i, (num_envs, env_config) in enumerate(remaining_configs):
            print(
                f"\nProgress: {i + 1}/{remaining_count} (Overall: {completed_count + i + 1}/{total_configs})"
            )
            self.results_df.append(self.benchmark_configuration(num_envs, env_config))
            self.results_df.to_csv()

    def plot_param_impact(self, save_filename: str | None = None, filter_params: Callable[[str, Any], bool] | None = None):
        """Plot comparison between params on steps per second to number of environments."""
        fig, ax = plt.subplots(figsize=(15, 12))
        fig.suptitle(
            "AtariVectorEnv Performance Overview",
            fontsize=16,
            fontweight="bold",
        )

        param_group = self.results_df.groupby("param")
        param_value_group = param_group.groupby("value")

        first_param = list(self.parameter_definitions.keys())[0]
        if first_param in df.columns:
            for param_value in sorted(df[first_param].unique()):
                subset = df[df[first_param] == param_value]
                label = (
                    f"{first_param}={param_value}"
                    if param_value != 0
                    else f"{first_param}=auto"
                )
                ax.errorbar(
                    subset["num_envs"],
                    subset["steps_per_second"],
                    yerr=subset["std_dev"],
                    label=label,
                    marker="o",
                    capsize=3,
                )
        ax.set_xlabel("Number of Environments")
        ax.set_ylabel("Steps per Second")
        ax.legend()

        plt.tight_layout()
        if save_filename:
            plt.savefig(save_filename)
        plt.show()

    def plot_resource_usage(self, filter_params: str):
        """Plot resource usage analysis."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle("Resource Usage Analysis", fontsize=14, fontweight="bold")

        # CPU usage vs performance
        ax1 = axes[0]
        scatter = ax1.scatter(
            self.results_df["cpu_usage"],
            self.results_df["steps_per_second"],
            c=self.results_df["num_envs"],
            cmap="viridis",
            alpha=0.7,
            s=50,
        )
        ax1.set_xlabel("CPU Usage (%)")
        ax1.set_ylabel("Steps per Second")
        ax1.set_title("CPU Usage vs Performance")
        ax1.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Number of Environments")

        # Memory usage vs performance
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            self.results_df["memory_usage"],
            self.results_df["steps_per_second"],
            c=self.results_df["num_envs"],
            cmap="plasma",
            alpha=0.7,
            s=50,
        )
        ax2.set_xlabel("Memory Usage (%)")
        ax2.set_ylabel("Steps per Second")
        ax2.set_title("Memory Usage vs Performance")
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label("Number of Environments")

        plt.tight_layout()

    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results_df:
            print("No results to summarize.")
            return

        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        # Find best performing configuration
        best_result = max(self.results_df, key=lambda r: r.steps_per_second)
        print(f"\nBest Performance:")
        print(f"  Configuration: {best_result.config}")
        print(
            f"  Performance: {best_result.steps_per_second:.1f} ± {best_result.std_dev:.1f} steps/sec"
        )
        print(f"  CPU Usage: {best_result.cpu_usage:.1f}%")
        print(f"  Memory Usage: {best_result.memory_usage:.1f}%")

        # Parameter impact analysis
        df = pd.DataFrame(
            [
                {
                    "num_envs": r.config.num_envs,
                    "batch_size": r.config.batch_size,
                    "autoreset_mode": r.config.autoreset_mode,
                    "num_threads": r.config.num_threads,
                    "performance": r.steps_per_second,
                }
                for r in self.results_df
            ]
        )


if __name__ == "__main__":
    testing_num_envs = [8, 16, 32, 64, 128, 256]
    testing_env_params = {
        "num_threads": [0],
        "batch_size": []
    }

    # Initialize benchmark
    benchmark = BenchmarkAtariVectorEnv(
        testing_num_envs,
        testing_env_params,
        game="breakout",
        warmup_steps=1000,
        measurement_time=10.0,
        num_runs=3,
    )

    # Run all benchmarks
    benchmark.run_all_benchmarks()

    # Create visualizations
    benchmark.plot_param_impact("benchmark-all.png")

    # Save results
    benchmark.save_results("atari_benchmark_results.csv")

    # Print summary
    benchmark.print_summary()

    print("\nBenchmark complete! Check the plots and CSV file for detailed results.")
