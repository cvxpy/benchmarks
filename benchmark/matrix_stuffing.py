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
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing


class ConeMatrixStuffingBench:

    def setup(self):
        m = 2000
        n = 2000
        A = np.random.randn(m, n)
        C = np.random.rand(m // 2)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self.problem = problem

    def time_compile_problem(self):
        cms = ConeMatrixStuffing()
        cms.apply(self.problem)


class ParamConeMatrixStuffing:

    def setup(self):
        m = 200
        n = 200
        A = cp.Parameter((m, n))
        C = cp.Parameter(m // 2)
        b = cp.Parameter(m)
        A.value = np.random.randn(m, n)
        C.value = np.random.rand(m // 2)
        b.value = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self.problem = problem

    def time_compile_problem(self):
        cms = ConeMatrixStuffing()
        cms.apply(self.problem)


class SmallMatrixStuffing:

    def setup(self):
        m = 200
        n = 200
        A = np.random.randn(m, n)
        C = np.random.rand(m // 2)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self.problem = problem

    def time_compile_problem(self):
        cms = ConeMatrixStuffing()
        cms.apply(self.problem)


class ParamSmallMatrixStuffing:

    def setup(self):
        m = 100
        n = 100
        A = cp.Parameter((m, n))
        C = cp.Parameter(m // 2)
        b = cp.Parameter(m)
        A.value = np.random.randn(m, n)
        C.value = np.random.rand(m // 2)
        b.value = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)
        self.problem = problem

    def time_compile_problem(self):
        cms = ConeMatrixStuffing()
        cms.apply(self.problem)


if __name__ == "__main__":
    cms = ConeMatrixStuffing()
    cms.setup()
    cms.time_compile_problem()

    param_cms = ParamConeMatrixStuffing()
    param_cms.setup()
    param_cms.time_compile_problem()

    sms = SmallMatrixStuffing()
    sms.setup()
    sms.time_compile_problem()

    psms = ParamSmallMatrixStuffing()
    psms.setup()
    psms.time_compile_problem()
