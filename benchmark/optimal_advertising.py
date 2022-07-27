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


class OptimalAdvertising:
    """
    Taken from https://stanford.edu/~boyd/papers/cvx_short_course.html
    """

    def setup(self):
        np.random.seed(1)
        m = 250
        n = 1000
        SCALE = 10000
        B = np.random.lognormal(mean=8, size=(m, 1)) + 10000
        B = 1000 * np.round(B / 1000)

        P_ad = np.random.uniform(size=(m, 1))
        P_time = np.random.uniform(size=(1, n))
        P = P_ad.dot(P_time)

        T = np.sin(np.linspace(-2 * np.pi / 2, 2 * np.pi - 2 * np.pi / 2, n)) * SCALE
        T += -np.min(T) + SCALE
        c = np.random.uniform(size=(m,))
        c *= 0.6 * T.sum() / c.sum()
        c = 1000 * np.round(c / 1000)
        R = np.array([np.random.lognormal(c.min() / c[i]) for i in range(m)])

        # Solve optimization problem
        D = cp.Variable((m, n))
        Si = [cp.minimum(R[i] * P[i, :] @ D[i, :].T, B[i]) for i in range(m)]
        objective = cp.Maximize(cp.sum(Si))
        constraints = [D >= 0, D.T @ np.ones(m) <= T, D @ np.ones(n) >= c]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    opt_advert = OptimalAdvertising()
    opt_advert.setup()
    opt_advert.time_compile_problem()
