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
import pickle

import cvxpy as cp

from benchmark.benchmark_base import Benchmark


class QP1611Benchmark(Benchmark):
    filename = os.path.join(Benchmark.data_folder, "QP1611.pickle")

    @staticmethod
    def name() -> str:
        return "QP issue 1611"

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        with open(QP1611Benchmark.filename, "rb") as f:
            problem = pickle.load(f)
        return problem

    @staticmethod
    def get_solver():
        return cp.ECOS_BB


if __name__ == "__main__":
    bench = QP1611Benchmark()
    bench.run_benchmark()
    bench.print_benchmark_results()
