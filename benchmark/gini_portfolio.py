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
import scipy.stats as st

import cvxpy as cp


class Yitzhaki:
    def setup(self):
        rs = np.random.RandomState(123)
        N = 50
        T = 150
        cov = rs.rand(N, N) * 1.5 - 0.5
        cov = cov @ cov.T/1000 + np.diag(rs.rand(N) * 0.7 + 0.3)/1000
        mean = np.zeros(N) + 1/1000
        returns = st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)
        assets = ['Asset ' + str(i) for i in range(1, N + 1)]
        D = np.array([]).reshape(0, len(assets))
        for j in range(0, returns.shape[0]-1):
            D = np.concatenate((D, returns[j+1:] - returns[j, :]), axis=0)

        d = cp.Variable((int(T * (T - 1) / 2), 1))
        w = cp.Variable((N, 1))
        constraints = []
        all_pairs_ret_diff = D @ w
        constraints += [d >= all_pairs_ret_diff,
                        d >= -all_pairs_ret_diff,
                        w >= 0,
                        cp.sum(w) == 1]
        risk = cp.sum(d) / ((T - 1) * T)
        objective = cp.Minimize(risk * 1000)
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class Murray:
    def setup(self):
        rs = np.random.RandomState(123)
        N = 50
        T = 350
        cov = rs.rand(N, N) * 1.5 - 0.5
        cov = cov @ cov.T/1000 + np.diag(rs.rand(N) * 0.7 + 0.3)/1000
        mean = np.zeros(N) + 1/1000
        returns = st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)

        d = cp.Variable((int(T * (T - 1) / 2), 1))
        w = cp.Variable((N, 1))
        constraints = []
        ret_w = cp.Variable((T, 1))
        constraints.append(ret_w == returns @ w)
        mat = np.zeros((d.shape[0], T))
        """
        We need to create a vector that has the following entries:
            ret_w[i] - ret_w[j]
        for j in range(T), for i in range(j+1, T).
        We do this by building a numpy array of mostly 0's and 1's.
        (It would be better to use SciPy sparse matrix objects.)
        """
        ell = 0
        for j in range(T):
            for i in range(j + 1, T):
                # write to mat so that (mat @ ret_w)[ell] == var_i - var_j
                mat[ell, i] = 1
                mat[ell, j] = -1
                ell += 1
        all_pairs_ret_diff = mat @ ret_w
        constraints += [d >= all_pairs_ret_diff,
                        d >= -all_pairs_ret_diff,
                        w >= 0,
                        cp.sum(w) == 1]
        risk = cp.sum(d) / ((T - 1) * T)
        objective = cp.Minimize(risk * 1000)
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class Cajas:
    def setup(self):
        rs = np.random.RandomState(123)
        N = 50
        T = 350
        cov = rs.rand(N,N) * 1.5 - 0.5
        cov = cov @ cov.T/1000 + np.diag(rs.rand(N) * 0.7 + 0.3)/1000
        mean = np.zeros(N) + 1/1000
        returns = st.multivariate_normal.rvs(mean=mean, cov=cov, size=T, random_state=rs)

        w = cp.Variable((N,1))
        constraints = []
        a = cp.Variable((T,1))
        b = cp.Variable((T,1))
        y = cp.Variable((T,1))
        owa_w = []
        for i in range(1,T+1):
            owa_w.append(2*i - 1 - T)
        owa_w = np.array(owa_w) / (T * (T-1))
        constraints = [returns @ w == y,
                       w >= 0,
                       cp.sum(w) == 1]
        for i in range(T):
            constraints += [a[i] + b >= cp.multiply(owa_w[i], y)]
        risk = cp.sum(a + b)
        objective = cp.Minimize(risk * 1000)
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == "__main__":
    yitzhaki = Yitzhaki()
    yitzhaki.setup()
    yitzhaki.time_compile_problem()
    print(f"compilation time: {yitzhaki.problem._compilation_time:.3f}")

    murray = Murray()
    murray.setup()
    murray.time_compile_problem()
    print(f"compilation time: {murray.problem._compilation_time:.3f}")

    cajas = Cajas()
    cajas.setup()
    cajas.time_compile_problem()
    print(f"compilation time: {cajas.problem._compilation_time:.3f}")
