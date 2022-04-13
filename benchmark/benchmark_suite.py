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
import re
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
        self._benchmarks = []
        self.cvxpy_version = cvxpy.__version__

    @property
    def benchmarks(self):
        assert self._benchmarks, "No benchmark has been registered yet."
        return self._benchmarks

    def register_benchmark(self, benchmark):
        assert isinstance(benchmark, Benchmark), \
            "The benchmark to register must be an instance of the 'Benchmark' base class."
        self._benchmarks.append(benchmark)

    def register_default_benchmarks(self) -> None:
        benchmarks = [
            CVaRBenchmark(),
            QP1611Benchmark(),
            SimpleLPBenchmark(),
            SimpleScalarParametrizedLPBenchmark(),
            SimpleFullyParametrizedLPBenchmark(),
            SDPSegfault1132Benchmark(),
        ]
        for b in benchmarks:
            self.register_benchmark(b)

    def run_benchmarks(self, repetitions: int = 1) -> Optional[str]:
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

        df_min, df_max, df_mean = (
            grouper.min(),
            grouper.max(),
            grouper.mean()
        )

        df_min_formatted = self.format_result(df_min, unit, normalize=self.normalize())
        df_max_formatted = self.format_result(df_max, unit, normalize=self.normalize())
        df_mean_formatted = self.format_result(df_mean, unit, normalize=self.normalize())

        output = f"Summary statistics of {len(dataframes)} runs:\n\n"
        output += "Best:\n\n"
        output += df_min_formatted.to_markdown(floatfmt=".2f")
        output += "\n\nWorst:\n\n"
        output += df_max_formatted.to_markdown(floatfmt=".2f")
        output += "\n\nMean:\n\n"
        output += df_mean_formatted.to_markdown(floatfmt=".2f") + "\n\n"

        return output

    def create_report(self, timings: str, memory: str, max_memory: float):
        src_path = pathlib.Path(__file__).parent.resolve()
        template_file = os.path.join(src_path, "benchmark_report_template.md")

        with open(template_file) as f:
            template = f.read()

        if self.normalize():
            time_normalization_message = (
                "Timings are normalized with respect to the fastest implementation.\n\n"
            )
            memory_normalization_message = (
                "Memory consumptions is normalized with respect"
                " to the most parsimonious implementation.\n\n"
            )
            timings = time_normalization_message + timings
            memory = memory_normalization_message + memory

        template = template.replace("COMPILATION_PLACEHOLDER", timings)
        template = template.replace("MEMORY_PLACEHOLDER", memory)

        template = template.replace("CVXPY_VERSION", self.cvxpy_version)
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

        version_1 = input(
            "Install/checkout the first version of CVXPY you want to test "
            "and type a name in this console"
        )
        assert len(version_1) > 0, "Please specify a version name through the input."

        repeated_version_1_timings, version_1_memory = \
            self.run_single_version_n_times(repetitions, version_1)

        version_2 = input(
            "Install/checkout the second version of CVXPY you want to test "
            "and type a name in this console"
        )
        assert len(version_2) > 0, "Please specify a version name through the input."

        repeated_version_2_timings, version_2_memory = \
            self.run_single_version_n_times(repetitions, version_2)

        repeated_timings = [
            pd.concat([feat, ref], axis=1)
            for feat, ref in zip(repeated_version_1_timings, repeated_version_2_timings)
        ]

        self.cvxpy_version = f"{version_1} / {version_2}"

        return repeated_timings, pd.concat([version_1_memory, version_2_memory], axis=1)

    def run_single_version_n_times(self, repetitions: int, version_name: str):
        repeated_timings = []
        memory_traces = None
        for repetition in range(1, repetitions + 1):
            print(f"\n\nRunning {version_name} {repetition=}/{repetitions} in subprocess...")
            timings, memory_traces = self.run_single_version()
            timings.name = version_name
            repeated_timings.append(timings)
            memory_traces.name = version_name
        assert memory_traces is not None
        return repeated_timings, memory_traces

    def run_single_version(self):

        timings = []
        memory_traces = []
        for bench in self.benchmarks:
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

            timing_regex = re.compile(r"(?<=^TIMING )\d+\.\d+")
            timing = float(timing_regex.findall(output[-3])[0])
            timings.append(timing)

            memory_regex = re.compile(r"(?<=^MEMORY )\d+\.\d+")
            memory_usage = float(memory_regex.findall(output[-2])[0])
            memory_traces.append(memory_usage)

        bench_names = [b.name() for b in self.benchmarks]
        return pd.Series(timings, bench_names), pd.Series(memory_traces, bench_names),

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

        timings = []
        memory_traces = []
        for bench in self.benchmarks:

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
