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

import cvxpy as cp
import numpy as np


class SimpleLPBenchmark:
    def setup(self):
        n = int(1e7)
        c = np.arange(n)
        x = cp.Variable(n)
        objective = cp.Minimize(c @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class SimpleFullyParametrizedLPBenchmark:
    def setup(self):
        n = int(1e4)
        p = cp.Parameter(n)
        x = cp.Variable(n)
        objective = cp.Minimize(p @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class SimpleScalarParametrizedLPBenchmark:
    def setup(self):
        n = int(1e6)
        p = cp.Parameter()
        c = np.arange(n)
        x = cp.Variable(n)
        objective = cp.Minimize((p * c) @ x)
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    simple_lp = SimpleLPBenchmark()
    simple_lp.setup()
    simple_lp.time_compile_problem()

    simple_fully_lp = SimpleFullyParametrizedLPBenchmark()
    simple_fully_lp.setup()
    simple_fully_lp.time_compile_problem()

    simple_scalar_lp = SimpleScalarParametrizedLPBenchmark()
    simple_scalar_lp.setup()
    simple_scalar_lp.time_compile_problem()
