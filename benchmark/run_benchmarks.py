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
import argparse

from benchmark.benchmark_suite import (CurrentVersionBenchmarkSuite,
                                       InteractiveComparisonBenchmarkSuite)


def run_benchmarks(
    repetitions: int = 1, mode: str = "current", output_file: str = "benchmark_report.md"
):

    if mode == "current":
        suite = CurrentVersionBenchmarkSuite(output_file)
    elif mode == "interactive":
        suite = InteractiveComparisonBenchmarkSuite(output_file)
    else:
        raise ValueError(f"Unknown benchmark mode {mode}")
    suite.run_benchmarks(repetitions)


if __name__ == "__main__":
    # Example:
    # python run_benchmarks.py --repetitions=3 --mode="current"

    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", default=1, type=int)
    parser.add_argument("--mode", default="current", type=str)
    args = parser.parse_args()
    run_benchmarks(args.repetitions, args.mode)
