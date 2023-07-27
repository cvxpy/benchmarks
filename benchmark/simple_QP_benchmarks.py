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


class SimpleQPBenchmark():
    def setup(self):
        m = 150
        n = 100
        p = 5
        P = np.random.randn(n, n)
        P = np.matmul(P.T, P)
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = np.matmul(G, np.random.randn(n))
        A = np.random.randn(p, n)
        b = np.random.randn(p)

        x = cp.Variable(n)
        problem = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + cp.matmul(q.T, x)),
                       [cp.matmul(G, x) <= h,
                       cp.matmul(A, x) == b])
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


class ParametrizedQPBenchmark():
    def setup(self):
        m = 150
        n = 100
        np.random.seed(1)
        A = cp.Parameter((m, n))
        b = cp.Parameter((m,))

        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A@x - b))
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


class LeastSquares():
    def setup(self):
        m = 5000
        n = 500
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum_squares(A @ x - b)
        problem = cp.Problem(cp.Minimize(cost))
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


if __name__ == '__main__':
    simple_qp = SimpleQPBenchmark()
    simple_qp.setup()
    simple_qp.time_compile_problem()
    print(f"compilation time: {simple_qp.problem._compilation_time:.3f}")

    param_qp = ParametrizedQPBenchmark()
    param_qp.setup()
    param_qp.time_compile_problem()
    print(f"compilation time: {param_qp.problem._compilation_time:.3f}")

    least_squares = LeastSquares()
    least_squares.setup()
    least_squares.time_compile_problem()
    print(f"compilation time: {least_squares.problem._compilation_time:.3f}")
