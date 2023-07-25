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


class TvInpainting:

    def setup(self):
        Uorig = np.random.randn(512, 512, 3)
        rows, cols, colors = Uorig.shape
        known = np.zeros((rows, cols, colors))
        for i in range(rows):
            for j in range(cols):
                if np.random.random() > 0.7:
                    for k in range(colors):
                        known[i, j, k] = 1

        Ucorr = known * Uorig  # This is elementwise mult on numpy arrays.
        variables = []
        constraints = []
        for i in range(colors):
                U = cp.Variable(shape=(rows, cols))
                variables.append(U)
                constraints.append(cp.multiply(
                    known[:, :, i], U) == cp.multiply(
                    known[:, :, i], Ucorr[:, :, i]))
        problem = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    prob = TvInpainting()
    prob.setup()
    prob.time_compile_problem()
