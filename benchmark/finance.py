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
import scipy.sparse as sp


class CVaRBenchmark:
    timeout = 999

    def setup(self):
        # Replaced real data with random values
        np.random.seed(0)
        price_scenarios = np.random.randn(131072, 192)
        forward_price_scenarios = np.random.randn(131072, 192)
        asset_energy_limits = np.random.randn(192, 2)
        bid_curve_prices = np.random.randn(192, 3)
        cvar_prob = 0.95
        cvar_kappa = 2.0

        num_scenarios, num_assets = price_scenarios.shape
        num_energy_segments = bid_curve_prices.shape[1] + 1

        price_segments = np.sum(
            forward_price_scenarios[:, :, None] > bid_curve_prices[None], axis=2
        )
        price_segments_flat = (
            price_segments + np.arange(num_assets) * num_energy_segments
        ).reshape(-1)
        price_segments_sp = sp.coo_matrix(
            (
                np.ones(num_scenarios * num_assets),
                (np.arange(num_scenarios * num_assets), price_segments_flat),
            ),
            shape=(num_scenarios * num_assets, num_assets * num_energy_segments),
        )

        prices_flat = (price_scenarios - forward_price_scenarios).reshape(-1)
        scenario_sum = sp.coo_matrix(
            (
                np.ones(num_scenarios * num_assets),
                (
                    np.repeat(np.arange(num_scenarios), num_assets),
                    np.arange(num_scenarios * num_assets),
                ),
            )
        )

        A = np.asarray((scenario_sum @ sp.diags(prices_flat) @ price_segments_sp).todense())
        c = np.mean(A, axis=0)
        gamma = 1.0 / (1.0 - cvar_prob) / num_scenarios
        kappa = cvar_kappa
        x_min = np.tile(asset_energy_limits[:, 0:1], (1, num_energy_segments)).reshape(-1)
        x_max = np.tile(asset_energy_limits[:, 1:2], (1, num_energy_segments)).reshape(-1)

        alpha = cp.Variable()
        x = cp.Variable(num_assets * num_energy_segments)

        problem = cp.Problem(
            cp.Minimize(c.T @ x),
            [
                alpha + gamma * cp.sum(cp.pos(A @ x - alpha)) <= kappa,
                x >= x_min,
                x <= x_max,
            ],
        )

        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


class FactorCovarianceModel:
    def setup(self):
        n = 14500
        m = 250
        np.random.seed(1)
        mu = np.abs(np.random.randn(n, 1))
        Sigma_tilde = np.random.randn(m, m)
        Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
        D = sp.diags(np.random.uniform(0, 0.9, size=n))
        F = np.random.randn(n, m)

        # Factor model portfolio optimization.
        w = cp.Variable(n)
        f = cp.Variable(m)
        self.gamma = cp.Parameter(nonneg=True)
        self.Lmax = cp.Parameter()
        ret = mu.T @ w
        risk = cp.quad_form(f, Sigma_tilde, assume_PSD=True) + cp.sum_squares(np.sqrt(D) @ w)
        objective = cp.Maximize(ret - self.gamma * risk)
        constraints = [cp.sum(w) == 1, f == F.T @ w, cp.norm(w, 1) <= self.Lmax]
        problem = cp.Problem(objective, constraints)
        self.problem = problem

    def time_compile_problem(self):
        self.Lmax.value = 2
        self.gamma.value = 0.1
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    cvar = CVaRBenchmark()
    cvar.setup()
    cvar.time_compile_problem()
    print(f"compilation time: {cvar.problem._compilation_time:.3f}")

    factor_covariance = FactorCovarianceModel()
    factor_covariance.setup()
    factor_covariance.time_compile_problem()
    print(f"compilation time: {factor_covariance.problem._compilation_time:.3f}")
