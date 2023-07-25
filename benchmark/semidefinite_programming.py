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


class SemidefiniteProgramming:
    """
    Taken from https://github.com/cvxpy/cvxpy/blob/master/examples/notebooks/WWW/sdp.ipynb
    """

    def setup(self):
        # Generate a random SDP.
        n = 125
        p = 75
        np.random.seed(1)
        C = np.random.randn(n, n)
        A = []
        b = []
        for i in range(p):
            A.append(np.random.randn(n, n))
            b.append(np.random.randn())

        # Define and solve the CVXPY problem.
        # Create a symmetric matrix variable.
        X = cp.Variable((n, n), symmetric=True)
        constraints = [X >> 0]
        constraints += [cp.trace(A[i] @ X) == b[i] for i in range(p)]
        objective = cp.Minimize(cp.trace(C @ X))
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class SemidefiniteExample:

    def setup(self):
        def randn_symm(n):
            A = np.random.randn(n, n)
            return (A + A.T) / 2

        def randn_psd(n):
            A = 1. / 10 * np.random.randn(n, n)
            return np.matmul(A, A.T)

        n = 100
        p = 100
        C = randn_psd(n)
        As = [randn_symm(n) for _ in range(p)]
        Bs = np.random.randn(p)

        X = cp.Variable((n, n), PSD=True)
        objective = cp.trace(cp.matmul(C, X))
        constraints = [
            cp.trace(cp.matmul(As[i], X)) == Bs[i] for i in range(p)]
        problem = cp.Problem(cp.Minimize(objective), constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == "__main__":
    sdp = SemidefiniteProgramming()
    sdp.setup()
    sdp.time_compile_problem()

    prob = SemidefiniteExample()
    prob.setup()
    prob.time_compile_problem()
