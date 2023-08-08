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


class SDPSegfault1132Benchmark:
    timeout = 999

    def setup(self):
        n = 100
        alpha = 1
        np.random.seed(0)
        points = np.random.rand(5, n)
        xtx = points.T @ points
        xtxd = np.diag(xtx)
        e = np.ones((n,))
        D = np.outer(e, xtxd) - 2 * xtx + np.outer(xtxd, e)
        # Construct W
        W = np.ones((n, n))

        # Define V and e
        n = D.shape[0]
        x = -1 / (n + np.sqrt(n))
        y = -1 / np.sqrt(n)
        V = np.ones((n, n - 1))
        V[0, :] *= y
        V[1:, :] *= x
        V[1:, :] += np.eye(n - 1)
        e = np.ones((n, 1))

        # Solve optimization problem
        G = cp.Variable((n - 1, n - 1), PSD=True)
        A = cp.kron(e, cp.reshape(cp.diag(V @ G @ V.T), (1, n)))
        B = cp.kron(e.T, cp.reshape(cp.diag(V @ G @ V.T), (n, 1)))
        C = alpha * cp.norm(cp.multiply(W, A + B - 2 * V @ G @ V.T - D), p="fro")
        objective = cp.Maximize(cp.trace(G) - C)
        problem = cp.Problem(objective, [])
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    sdp_1132 = SDPSegfault1132Benchmark()
    sdp_1132.setup()
    sdp_1132.time_compile_problem()
