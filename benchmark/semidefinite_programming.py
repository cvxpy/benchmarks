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

import numpy as np

import cvxpy as cp


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


if __name__ == "__main__":
    sdp = SemidefiniteProgramming()
    sdp.setup()
    sdp.time_compile_problem()
    print(f"compilation time: {sdp.problem._compilation_time:.3f}")
