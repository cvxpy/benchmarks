"""
Copyright, the CVXPY authors
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import pathlib
import subprocess
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

import cvxpy
import pandas as pd

from benchmark.benchmark_base import Benchmark
from benchmark.cvar_benchmark import CVaRBenchmark
from benchmark.qp_1611_benchmark import QP1611Benchmark
from benchmark.sdp_segfault_1132_benchmark import SDPSegfault1132Benchmark
from benchmark.simple_LP_benchmarks import (
    SimpleFullyParametrizedLPBenchmark, SimpleLPBenchmark,
    SimpleScalarParametrizedLPBenchmark)


class BenchmarkSuite(ABC):
    def __init__(self, output_file: str = "benchmark_report.md"):
        # If output_file is an empty string the report is returned as a string

        self.start_time = None
        self.end_time = None
        self.output_file = output_file

    def get_registered_benchmarks(self) -> List[Benchmark]:
        benchmarks = [
            CVaRBenchmark(),
            QP1611Benchmark(),
            SimpleLPBenchmark(),
            SimpleScalarParametrizedLPBenchmark(),
            SimpleFullyParametrizedLPBenchmark(),
            SDPSegfault1132Benchmark(),
        ]
        return benchmarks

    def run_benchmarks(self, repetitions: int) -> Optional[str]:
        assert repetitions > 0

        self.start_time = time.time()

        repeated_timings, memory_traces = self.get_timings_and_memory_traces(repetitions)

        self.end_time = time.time()
        print("Finished running benchmarks, generating report.")

        if repetitions == 1:
            report = self.create_single_report(repeated_timings[0], memory_traces)
        else:
            report = self.create_multiple_report(repeated_timings, memory_traces)

        print("Report generated successfully.")

        if self.output_file:
            with open(self.output_file, "w") as f:
                f.write(report)
        else:
            return report

    def create_single_report(self, timings: pd.DataFrame, memory_traces: pd.DataFrame) -> str:
        timings_formatted = self.format_result(
            timings, unit="s", normalize=self.normalize()
        ).to_markdown(floatfmt=".2f")
        memory_traces_formatted = self.format_result(
            memory_traces, unit="GiB", normalize=self.normalize()
        ).to_markdown(floatfmt=".2f")
        return self.create_report(
            timings_formatted, memory_traces_formatted, memory_traces.max().max()
        )

    @staticmethod
    def format_result(df: pd.DataFrame, unit: str, normalize: bool) -> pd.DataFrame:
        if normalize:
            min_values = df.min(axis=1)
            df_normalized = df.div(min_values, axis=0)
            df_formatted = df_normalized.sort_index(axis=1).sort_index(axis=0)
            df_formatted.insert(len(df.columns), "Reference", min_values)
            df_formatted = df_formatted.applymap(lambda x: f"{x:.2f}")
            df_formatted["Reference"] += unit
            return df_formatted
        else:
            return df.sort_index(axis=0).applymap(lambda x: f"{x:.2f}") + unit

    def create_multiple_report(
        self, repeated_timings: List[pd.DataFrame], memory_traces: pd.DataFrame
    ) -> str:
        timings_output = self.format_multiple_dataframes(repeated_timings, unit="s")
        memory_traces_formatted = self.format_result(
            memory_traces, unit="GiB", normalize=self.normalize()
        ).to_markdown(floatfmt=".2f")
        return self.create_report(
            timings_output, memory_traces_formatted, memory_traces.max().max()
        )

    def format_multiple_dataframes(self, dataframes: List[pd.DataFrame], unit):
        df_repeated_timings = pd.concat(dataframes)
        grouper = df_repeated_timings.groupby(df_repeated_timings.index)

        df_min, df_max, df_mean, df_std = (
            grouper.min(),
            grouper.max(),
            grouper.mean(),
            grouper.std(),
        )

        df_min_formatted = self.format_result(df_min, unit, normalize=self.normalize())
        df_max_formatted = self.format_result(df_max, unit, normalize=self.normalize())
        df_mean_formatted = self.format_result(df_mean, unit, normalize=self.normalize())

        if self.normalize():
            # normalize standard deviation w.r.t. to mean normalization factor
            mean_min_values = df_mean.min(axis=1)
            mean_normalized = df_mean.div(mean_min_values, axis=0)
            df_std = df_std * (mean_normalized.div(df_mean))
        std_formatted = self.format_result(df_std, "", normalize=False)

        output = f"Summary statistics of {len(dataframes)} runs:\n\n"
        output += "Best:\n\n"
        output += df_min_formatted.to_markdown(floatfmt=".2f")
        output += "\n\nWorst:\n\n"
        output += df_max_formatted.to_markdown(floatfmt=".2f")
        output += "\n\nMean (SD):\n\n"
        output += (df_mean_formatted + " (" + std_formatted + ")").to_markdown() + "\n\n"

        return output

    def create_report(self, timings: str, memory: str, max_memory: float):
        src_path = pathlib.Path(__file__).parent.resolve()
        template_file = os.path.join(src_path, "benchmark_report_template.md")

        with open(template_file) as f:
            template = f.read()

        if self.normalize():
            time_normalization_message = (
                "Timings are normalized with respect to the" " fastest implementation.\n\n"
            )
            memory_normalization_message = (
                "Memory consumptions is normalized with respect"
                " to the most parsimonious implementation.\n\n"
            )
            timings = time_normalization_message + timings
            memory = memory_normalization_message + memory

        template = template.replace("COMPILATION_PLACEHOLDER", timings)
        template = template.replace("MEMORY_PLACEHOLDER", memory)

        template = template.replace("CVXPY_VERSION", cvxpy.__version__)
        template = template.replace("DATE", datetime.now().isoformat())
        template = template.replace("TOTAL_TIME", f"{self.end_time - self.start_time:.2f}s")
        template = template.replace("MAX_MEMORY", f"{max_memory:.2f}GiB")

        return template

    @abstractmethod
    def get_timings_and_memory_traces(
        self, repetitions: int
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        pass

    @staticmethod
    @abstractmethod
    def normalize() -> bool:
        pass


class InteractiveComparisonBenchmarkSuite(BenchmarkSuite):
    def get_timings_and_memory_traces(
        self, repetitions: int
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:

        input(
            "Checkout the feature branch in a different terminal window. "
            "Press any key to continue."
        )

        repeated_feature_timings = []
        feature_memory_traces = None
        for repetition in range(1, repetitions + 1):
            print(f"\n\nRunning feature {repetition=}/{repetitions} in subprocess...")
            feature_timings, feature_memory_traces = self.single_branch_run()
            feature_timings.name = "FEATURE"
            repeated_feature_timings.append(feature_timings)
            feature_memory_traces.name = "FEATURE"
        assert feature_memory_traces is not None

        input(
            "Checkout the reference branch in a different terminal window. "
            "Press any key to continue."
        )

        repeated_reference_timings = []
        reference_memory_traces = None
        for repetition in range(1, repetitions + 1):
            print(f"\n\nRunning reference {repetition=}/{repetitions} in subprocess...")
            reference_timings, reference_memory_traces = self.single_branch_run()
            reference_timings.name = "REFERENCE"
            repeated_reference_timings.append(reference_timings)
            reference_memory_traces.name = "REFERENCE"
        assert reference_memory_traces is not None

        repeated_timings = [
            pd.concat([feat, ref], axis=1)
            for feat, ref in zip(repeated_feature_timings, repeated_reference_timings)
        ]

        return repeated_timings, pd.concat([feature_memory_traces, reference_memory_traces], axis=1)

    def single_branch_run(self):
        benchs = self.get_registered_benchmarks()

        timings = []
        memory_traces = []
        for bench in benchs:
            # Switching branches requires a dynamic reload of modules,
            # which is achieved by running the benchmark in
            # a subprocess, creating a suboptimal interface here.
            command = (
                f"python -c "
                f"'from {bench.__class__.__module__} import {bench.__class__.__name__}; "
                f"bench={bench.__class__.__name__}(); "
                f"bench.run_benchmark(); "
                f'print("TIMING", bench.timing); '
                f'print("MEMORY", bench.memory_peak)\''
            )
            output = subprocess.check_output(command, shell=True).decode("utf-8").split("\n")
            timings.append(float(next(iter([x.split(" ")[-1] for x in output if "TIMING" in x]))))
            memory_traces.append(
                float(next(iter([x.split(" ")[-1] for x in output if "MEMORY" in x])))
            )

        bench_names = [b.name() for b in benchs]
        return pd.Series(timings, bench_names), pd.Series(memory_traces, bench_names)

    @staticmethod
    def normalize():
        return True


class CurrentVersionBenchmarkSuite(BenchmarkSuite):
    def get_timings_and_memory_traces(
        self, repetitions: int
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        repeated_timings = []
        memory_traces = None
        for repetition in range(1, repetitions + 1):
            print(f"\n\nRunning {repetition=}/{repetitions}...")
            timings, memory_traces = self.single_run()
            memory_traces = memory_traces
            repeated_timings.append(timings)
        assert memory_traces is not None
        return repeated_timings, memory_traces

    def single_run(self):
        benchs = self.get_registered_benchmarks()

        timings = []
        memory_traces = []
        for bench in benchs:

            bench_timings = {}
            bench_memory = {}

            bench.run_benchmark()
            bench_timings["CurrentVersion"] = bench.timing
            bench_memory["CurrentVersion"] = bench.memory_peak

            timings.append(pd.Series(bench_timings, name=bench.name()))
            memory_traces.append(pd.Series(bench_memory, name=bench.name()))

        return pd.concat(timings, axis=1).T, pd.concat(memory_traces, axis=1).T

    @staticmethod
    def normalize() -> bool:
        return False
