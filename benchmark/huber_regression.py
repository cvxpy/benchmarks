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


class HuberRegression():
    """
    Taken from https://stanford.edu/~boyd/papers/cvx_short_course.html
    """

    def setup(self):
        np.random.seed(1)
        n = 3000
        p = 0.12
        SAMPLES = int(1.5 * n)
        beta_true = 5 * np.random.normal(size=(n, 1))
        X = np.random.randn(n, SAMPLES)
        Y = np.zeros((SAMPLES, 1))
        v = np.random.normal(size=(SAMPLES, 1))

        # Generate the sign changes.
        factor = 2 * np.random.binomial(1, 1 - p, size=(SAMPLES, 1)) - 1
        Y = factor * X.T.dot(beta_true) + v
        beta = cp.Variable((n, 1))
        # Solve a huber regression problem
        cost = cp.sum(cp.huber(X.T @ beta - Y, 1))
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    huber_regression = HuberRegression()
    huber_regression.setup()
    huber_regression.time_compile_problem()
