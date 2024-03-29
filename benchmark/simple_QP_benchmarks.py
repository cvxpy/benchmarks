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
from scipy.linalg import dft


class SimpleQPBenchmark:
    def setup(self):
        m = 2000
        n = 400
        p = 5
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = np.matmul(P.T, P)
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = np.matmul(G, np.random.randn(n))
        A = np.random.randn(p, n)
        b = np.random.randn(p)

        x = cp.Variable(n)
        problem = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, P, assume_PSD=True) +
                                         cp.matmul(q.T, x)),
                             [cp.matmul(G, x) <= h,
                              cp.matmul(A, x) == b])
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


class ParametrizedQPBenchmark:
    def setup(self):
        m = 250
        n = 100
        np.random.seed(1)
        A = cp.Parameter((m, n))
        b = cp.Parameter((m,))

        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        constraints = [0 <= x, x <= 1]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


class LeastSquares:
    def setup(self):
        m = 5000
        n = 1000
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum_squares(A @ x - b)
        problem = cp.Problem(cp.Minimize(cost))
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.OSQP)


class UnconstrainedQP:
    """
    Related issue: https://github.com/cvxpy/cvxpy/issues/2205
    """
    def setup(self):
        N_r = 16
        N_t = 2
        N_s = 6

        x = np.random.randint(2, size=N_s * N_r * N_t)

        H = dft(N_s * N_r * N_t) * 1j
        H_H = H.conj().T

        err = np.random.random(N_r)
        Err = np.kron(np.diag(np.ones(N_s * N_t)), np.diag(err))

        y = H_H @ Err @ H @ x

        var = cp.Variable(shape=(N_r), complex=True)
        Err_est = cp.kron(np.diag(np.ones(N_s * N_t)), cp.diag(var))

        res = cp.norm2(H_H @ Err_est @ H @ x - y)

        problem = cp.Problem(cp.Minimize(res))
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


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

    unconstrained = UnconstrainedQP()
    unconstrained.setup()
    unconstrained.time_compile_problem()
    print(f"compilation time: {unconstrained.problem._compilation_time:.3f}")
