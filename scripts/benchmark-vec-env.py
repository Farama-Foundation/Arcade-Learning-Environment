#!/usr/bin/env python3
"""
Comprehensive benchmarking script for AtariVectorEnv performance testing.
Tests the impact of num_envs, batch_size, autoreset_mode, and num_threads on steps per second.
"""
from __future__ import annotations

import dataclasses
import gc
import os
import time
from argparse import ArgumentParser
from typing import Any, Callable

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
    env_param_meaning: str

    mean_steps_per_second: Any
    std_steps_per_second: Any
    mean_cpu_usage: Any
    mean_memory_usage: Any


class BenchmarkAtariVectorEnv:
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
        avg_memory = (mem_start + mem_end) / 2

        envs.close()
        gc.collect()

        return steps_per_second, avg_cpu, avg_memory

    def benchmark_configuration(
        self, num_envs: int, env_params: dict[str, Any], env_param_meaning: str
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
            env_param_meaning=env_param_meaning,
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
            (row["num_envs"], row["env_params_meaning"], row["env_params"])
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
                        data=dataclasses.asdict(benchmark_result),
                        columns=self.results_df.columns,
                    ),
                )
            )

            self.results_df.to_csv(self.results_filename, index=False)

    def plot_steps_per_second(
        self,
        parameter_key: str = "num_envs",
        title: str | None = None,
        save_path: str | None = None,
        figsize: tuple = (10, 6),
        color: str = "skyblue",
    ) -> plt.Figure:
        """
        Plot steps per second with error bars from BenchmarkResult data.

        Args:
            results: List of BenchmarkResult objects or DataFrame with benchmark results
            parameter_key: Key to use for x-axis grouping ('num_envs' or key from env_params)
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
            color: Bar color

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Extract parameter values and sort
        if parameter_key == "num_envs":
            param_values = self.results_df["num_envs"].values
            param_label = "Number of Environments"
        else:
            # Extract from env_params
            param_values = [
                row["env_params"].get(parameter_key, "")
                for _, row in self.results_df.iterrows()
            ]
            param_label = parameter_key.replace("_", " ").title()
            # Use env_param_meaning if available and consistent
            if self.results_df["env_param_meaning"].nunique() == 1:
                param_label = self.results_df["env_param_meaning"].iloc[0]

        # Sort by parameter values
        sort_idx = np.argsort(param_values)
        param_values_sorted = np.array(param_values)[sort_idx]
        means = self.results_df["mean_steps_per_second"].values[sort_idx]
        stds = self.results_df["std_steps_per_second"].values[sort_idx]

        # Create the plot with error bars
        x_pos = range(len(param_values_sorted))
        ax.bar(
            x_pos,
            means,
            yerr=stds,
            capsize=5,
            alpha=0.7,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

        # Customize the plot
        ax.set_xlabel(param_label)
        ax.set_ylabel("Steps per Second")
        ax.set_title(title if title else f"Steps per Second vs {param_label}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(
            [str(pv) for pv in param_values_sorted],
            rotation=45 if len(param_values_sorted) > 5 else 0,
        )
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(
                i,
                mean + std + max(means) * 0.01,
                f"{mean:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_resource_usage(
        self,
        parameter_key: str = "num_envs",
        resource_types: list[str] = ["cpu_usage", "memory_usage"],
        title: str | None = None,
        save_path: str | None = None,
        figsize: tuple = (12, 8),
        colors: list[str] | None = None,
    ) -> plt.Figure:
        """
        Plot resource usage metrics from BenchmarkResult data.

        Args:
            results: List of BenchmarkResult objects or DataFrame with benchmark results
            parameter_key: Key to use for x-axis grouping ('num_envs' or key from env_params)
            resource_types: List of resource metrics to plot ('cpu_usage', 'memory_usage')
            title: Plot title (optional)
            save_path: Path to save the plot (optional)
            figsize: Figure size tuple
            colors: List of colors for different metrics (optional)

        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(len(resource_types), 1, figsize=figsize, sharex=True)

        # Handle single subplot case
        if len(resource_types) == 1:
            axes = [axes]

        # Default colors if not provided
        if not colors:
            colors = plt.cm.Set1(np.linspace(0, 1, len(resource_types)))

        # Extract parameter values and sort
        if parameter_key == "num_envs":
            param_values = self.results_df["num_envs"].values
            param_label = "Number of Environments"
        else:
            # Extract from env_params
            param_values = [
                row["env_params"].get(parameter_key, "")
                for _, row in self.results_df.iterrows()
            ]
            param_label = parameter_key.replace("_", " ").title()
            # Use env_param_meaning if available and consistent
            if self.results_df["env_param_meaning"].nunique() == 1:
                param_label = self.results_df["env_param_meaning"].iloc[0]

        # Sort by parameter values
        sort_idx = np.argsort(param_values)
        param_values_sorted = np.array(param_values)[sort_idx]
        x_pos = range(len(param_values_sorted))

        for idx, resource_type in enumerate(resource_types):
            ax = axes[idx]

            # Map resource type to dataframe column
            if resource_type == "cpu_usage":
                column_name = "mean_cpu_usage"
            elif resource_type == "memory_usage":
                column_name = "mean_memory_usage"
            else:
                column_name = f"mean_{resource_type}"

            if column_name not in self.results_df.columns:
                print(f"Warning: Column {column_name} not found in data")
                continue

            means = self.results_df[column_name].values[sort_idx]

            # Create line plot (no error bars for resource usage since we only have means)
            ax.plot(
                x_pos,
                means,
                marker="o",
                linestyle="-",
                linewidth=2,
                markersize=6,
                color=colors[idx],
                label=resource_type.replace("_", " ").title(),
            )

            # Fill area under the curve for better visualization
            ax.fill_between(x_pos, 0, means, alpha=0.2, color=colors[idx])

            # Customize subplot
            ylabel = resource_type.replace("_", " ").title()
            if "usage" in resource_type:
                ylabel += " (%)"
            elif "memory" in resource_type:
                ylabel += " (MB)"

            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")

            # Add value annotations
            for i, mean in enumerate(means):
                if mean > 0:  # Only annotate non-zero values
                    ax.annotate(
                        f"{mean:.1f}",
                        (i, mean),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=8,
                    )

        # Set x-axis labels on the bottom subplot
        axes[-1].set_xlabel(param_label)
        axes[-1].set_xticks(x_pos)
        axes[-1].set_xticklabels(
            [str(pv) for pv in param_values_sorted],
            rotation=45 if len(param_values_sorted) > 5 else 0,
        )

        # Set overall title
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        else:
            fig.suptitle(f"Resource Usage vs {param_label}", fontsize=14, y=0.98)

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
