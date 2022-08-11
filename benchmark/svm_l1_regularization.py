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


class SVMWithL1Regularization():
    """
    Taken from https://stanford.edu/~boyd/papers/cvx_short_course.html
    """

    def setup(self):
        np.random.seed(1)
        n = 500
        m = 25000
        DENSITY = 0.2
        beta_true = np.random.randn(n, 1)
        idxs = np.random.choice(range(n), int((1 - DENSITY) * n), replace=False)
        for idx in idxs:
            beta_true[idx] = 0

        offset = 0
        sigma = 45
        X = np.random.normal(0, 5, size=(m, n))
        Y = np.sign(X.dot(beta_true) + offset + np.random.normal(0, sigma, size=(m, 1)))

        # Solve optimization problem
        beta = cp.Variable((n, 1))
        v = cp.Variable()
        loss = cp.sum(cp.pos(1 - cp.multiply(Y, X @ beta - v)))
        reg = cp.norm(beta, 1)
        lambd = cp.Parameter(nonneg=True)
        objective = cp.Minimize(loss / m + lambd * reg)
        problem = cp.Problem(objective)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    svm_l1 = SVMWithL1Regularization()
    svm_l1.setup()
    svm_l1.time_compile_problem()
