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


class SlowPruningBenchmark():

    def setup(self):
        """Regression test for https://github.com/cvxpy/cvxpy/issues/1668

        Pruning matrices caused order-of-magnitude slow downs in compilation times.
        """
        s = 2000
        t = 10
        x = np.linspace(-100.0, 100.0, s)
        rows = 50
        var = cp.Variable(shape=(rows, t))

        cost = cp.sum_squares(
            var @ np.tile(np.array([x]), t).reshape((t, x.shape[0]))
            - np.tile(x, rows).reshape((rows, s))
        )
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.ECOS)


if __name__ == '__main__':
    issue_1668 = SlowPruningBenchmark()
    issue_1668.setup()
    issue_1668.time_compile_problem()
