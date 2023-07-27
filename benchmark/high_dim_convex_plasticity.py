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
from scipy.sparse import block_diag

import cvxpy as cp


class ConvexPlasticity:
    """
    Taken from https://github.com/andrash/convex-plasticity
    """

    def setup(self):
        E_dim = 70e3  # [Pa]
        E = 70e3 / E_dim  # [-] Young's modulus
        nu = 0.3  # [-] Poisson's ratio
        sig0 = 250 / E_dim  # [-] yield strength
        Et = E / 100.0  # [-] tangent modulus
        H = E * Et / (E - Et)  # [-] hardening modulus

        l, m = E * nu / (1 + nu) / (1 - 2 * nu), E / 2 / (1 + nu)  # noqa: E741

        # Elasticity stiffness matrix
        C = np.array(
            [
                [l + 2 * m, l, l, 0],
                [l, l + 2 * m, l, 0],
                [l, l, l + 2 * m, 0],
                [0, 0, 0, 2 * m],
            ]
        )
        S = np.linalg.inv(C)

        def criterion(sig: cp.Variable, p: cp.Variable):
            N = p.size
            dev = np.array(
                [
                    [2 / 3.0, -1 / 3.0, -1 / 3.0, 0],
                    [-1 / 3.0, 2 / 3.0, -1 / 3.0, 0],
                    [-1 / 3.0, -1 / 3.0, 2 / 3.0, 0],
                    [0, 0, 0, 1.0],
                ]
            )

            sig0_vec = np.repeat(sig0, N)
            s = dev @ sig
            return [np.sqrt(3 / 2) * cp.norm(s, axis=0) <= sig0_vec + p * H]

        particular_deps = np.array(
            [7.24276667e-04, -3.41842340e-04, 0.00000000e00, -2.65071262e-05]
        )
        deps_dummy = np.full((4000, 4), particular_deps)

        N = 500
        deps_local_values = deps_dummy[:N]

        deps = cp.Parameter((4, N), name="deps")
        sig_old = cp.Parameter((4, N), name="sig_old")
        sig_elas = sig_old + C @ deps
        sig = cp.Variable((4, N), name="sig")
        p_old = cp.Parameter((N,), nonneg=True, name="p_old")
        p = cp.Variable((N,), nonneg=True, name="p")

        sig_old.value = np.zeros((4, N))
        deps.value = np.zeros((4, N))
        p_old.value = np.zeros((N,))
        deps.value = deps_local_values.T

        delta_sig = sig - sig_elas
        D = H * np.eye(N)
        S_sparsed = block_diag([S for _ in range(N)])
        delta_sig_vector = cp.reshape(delta_sig, (N * 4))

        elastic_energy = cp.quad_form(delta_sig_vector, S_sparsed, assume_PSD=True)
        target_expression = 0.5 * elastic_energy + 0.5 * cp.quad_form(p - p_old, D)
        constraints = criterion(sig, p)

        problem = cp.Problem(cp.Minimize(target_expression), constraints)

        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == "__main__":
    plasticity = ConvexPlasticity()
    plasticity.setup()
    plasticity.time_compile_problem()

    print(f"compilation time: {plasticity.problem._compilation_time:.3f}")
