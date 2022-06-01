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
import pickle

import cvxpy as cp

class QP1611Benchmark():
    filename = os.path.join(os.path.join(pathlib.Path(__file__).parent.resolve(), "benchmark_data"), "QP1611.pickle")

    def setup(self):
        with open(QP1611Benchmark.filename, "rb") as f:
            problem = pickle.load(f)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.ECOS_BB)
