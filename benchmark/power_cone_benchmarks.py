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

Benchmarks for power cone atoms comparing approx=True vs approx=False.

- approx=True: Uses SOC approximation via rational decomposition
- approx=False: Uses native power cones (requires solver support)

These benchmarks measure compilation and solve times for:
- power(x, p) with various exponents
- geo_mean(x, p) with various weight configurations
- pnorm(x, p) with various p values
"""

import cvxpy as cp
import numpy as np


# =============================================================================
# Power Atom Benchmarks
# =============================================================================

class PowerApproxTrue:
    """Power atom with approx=True (SOC approximation)."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=1.5 (convex, non-integer)
        objective = cp.Minimize(cp.sum(c @ self.x))
        constraints = [
            cp.sum(cp.power(self.x, 1.5, approx=True)) <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PowerApproxFalse:
    """Power atom with approx=False (native power cones)."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=1.5 (convex, non-integer)
        objective = cp.Minimize(cp.sum(c @ self.x))
        constraints = [
            cp.sum(cp.power(self.x, 1.5, approx=False)) <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PowerNegativeExponentApproxTrue:
    """Power atom with negative exponent, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=-0.5 (convex, negative exponent)
        objective = cp.Minimize(cp.sum(cp.power(self.x, -0.5, approx=True)))
        constraints = [
            c @ self.x >= 10,
            self.x >= 0.01,
            self.x <= 10,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PowerNegativeExponentApproxFalse:
    """Power atom with negative exponent, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=-0.5 (convex, negative exponent)
        objective = cp.Minimize(cp.sum(cp.power(self.x, -0.5, approx=False)))
        constraints = [
            c @ self.x >= 10,
            self.x >= 0.01,
            self.x <= 10,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PowerFractionalExponentApproxTrue:
    """Power atom with fractional exponent (0 < p < 1), approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=0.3 (concave)
        objective = cp.Maximize(cp.sum(cp.power(self.x, 0.3, approx=True)))
        constraints = [
            c @ self.x <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PowerFractionalExponentApproxFalse:
    """Power atom with fractional exponent (0 < p < 1), approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        # Power with p=0.3 (concave)
        objective = cp.Maximize(cp.sum(cp.power(self.x, 0.3, approx=False)))
        constraints = [
            c @ self.x <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


# =============================================================================
# Geometric Mean Benchmarks
# =============================================================================

class GeoMeanApproxTrue:
    """Geometric mean with approx=True (SOC approximation)."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=True))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class GeoMeanApproxFalse:
    """Geometric mean with approx=False (native power cones)."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=False))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class GeoMeanLargeApproxTrue:
    """Large geometric mean with approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 500
        m = 250
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=True))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class GeoMeanLargeApproxFalse:
    """Large geometric mean with approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 500
        m = 250
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=False))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class GeoMeanWeightedApproxTrue:
    """Weighted geometric mean with approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        # Non-uniform positive weights
        weights = np.random.rand(n) + 0.5
        weights = weights / weights.sum()

        objective = cp.Maximize(cp.geo_mean(self.x, p=weights, approx=True))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class GeoMeanWeightedApproxFalse:
    """Weighted geometric mean with approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        # Non-uniform positive weights
        weights = np.random.rand(n) + 0.5
        weights = weights / weights.sum()

        objective = cp.Maximize(cp.geo_mean(self.x, p=weights, approx=False))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


# =============================================================================
# P-norm Benchmarks
# =============================================================================

class PnormApproxTrue:
    """P-norm with p=1.5, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        m = 2500
        self.x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        objective = cp.Minimize(cp.pnorm(self.x, p=1.5, approx=True))
        constraints = [
            A @ self.x == b,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PnormApproxFalse:
    """P-norm with p=1.5, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        m = 2500
        self.x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        objective = cp.Minimize(cp.pnorm(self.x, p=1.5, approx=False))
        constraints = [
            A @ self.x == b,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PnormP3ApproxTrue:
    """P-norm with p=3, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        m = 2500
        self.x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        objective = cp.Minimize(cp.pnorm(self.x, p=3, approx=True))
        constraints = [
            A @ self.x == b,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PnormP3ApproxFalse:
    """P-norm with p=3, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 5000
        m = 2500
        self.x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        objective = cp.Minimize(cp.pnorm(self.x, p=3, approx=False))
        constraints = [
            A @ self.x == b,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PnormFractionalApproxTrue:
    """P-norm with p=0.5, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        m = 500
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10
        c = np.random.rand(n)

        # Minimize linear, constrain pnorm
        objective = cp.Minimize(c @ self.x)
        constraints = [
            cp.pnorm(self.x, p=0.5, approx=True) >= 10,
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


class PnormFractionalApproxFalse:
    """P-norm with p=0.5, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        m = 500
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10
        c = np.random.rand(n)

        # Minimize linear, constrain pnorm
        objective = cp.Minimize(c @ self.x)
        constraints = [
            cp.pnorm(self.x, p=0.5, approx=False) >= 10,
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.CLARABEL)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.CLARABEL, verbose=False)


# =============================================================================
# SCS Solver Benchmarks
# =============================================================================

class PowerSCSApproxTrue:
    """Power atom with SCS solver, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        objective = cp.Minimize(cp.sum(c @ self.x))
        constraints = [
            cp.sum(cp.power(self.x, 1.5, approx=True)) <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.SCS, verbose=False)


class PowerSCSApproxFalse:
    """Power atom with SCS solver, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 1000
        self.x = cp.Variable(n, pos=True)
        c = np.random.rand(n) + 0.1

        objective = cp.Minimize(cp.sum(c @ self.x))
        constraints = [
            cp.sum(cp.power(self.x, 1.5, approx=False)) <= 100,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.SCS, verbose=False)


class GeoMeanSCSApproxTrue:
    """Geometric mean with SCS solver, approx=True."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=True))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.SCS, verbose=False)


class GeoMeanSCSApproxFalse:
    """Geometric mean with SCS solver, approx=False."""

    def setup(self):
        np.random.seed(42)
        n = 100
        m = 50
        self.x = cp.Variable(n, pos=True)
        A = np.random.rand(m, n) + 0.1
        b = np.random.rand(m) * 10

        objective = cp.Maximize(cp.geo_mean(self.x, approx=False))
        constraints = [
            A @ self.x <= b,
            self.x >= 0.01,
        ]
        self.problem = cp.Problem(objective, constraints)

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)

    def time_solve_problem(self):
        self.problem.solve(solver=cp.SCS, verbose=False)


if __name__ == '__main__':
    import time

    benchmarks = [
        # Power atom benchmarks
        ("PowerApproxTrue", PowerApproxTrue),
        ("PowerApproxFalse", PowerApproxFalse),
        ("PowerNegativeExponentApproxTrue", PowerNegativeExponentApproxTrue),
        ("PowerNegativeExponentApproxFalse", PowerNegativeExponentApproxFalse),
        ("PowerFractionalExponentApproxTrue", PowerFractionalExponentApproxTrue),
        ("PowerFractionalExponentApproxFalse", PowerFractionalExponentApproxFalse),
        # Geometric mean benchmarks
        ("GeoMeanApproxTrue", GeoMeanApproxTrue),
        ("GeoMeanApproxFalse", GeoMeanApproxFalse),
        ("GeoMeanLargeApproxTrue", GeoMeanLargeApproxTrue),
        ("GeoMeanLargeApproxFalse", GeoMeanLargeApproxFalse),
        ("GeoMeanWeightedApproxTrue", GeoMeanWeightedApproxTrue),
        ("GeoMeanWeightedApproxFalse", GeoMeanWeightedApproxFalse),
        # P-norm benchmarks
        ("PnormApproxTrue", PnormApproxTrue),
        ("PnormApproxFalse", PnormApproxFalse),
        ("PnormP3ApproxTrue", PnormP3ApproxTrue),
        ("PnormP3ApproxFalse", PnormP3ApproxFalse),
        ("PnormFractionalApproxTrue", PnormFractionalApproxTrue),
        ("PnormFractionalApproxFalse", PnormFractionalApproxFalse),
        # SCS solver benchmarks
        ("PowerSCSApproxTrue", PowerSCSApproxTrue),
        ("PowerSCSApproxFalse", PowerSCSApproxFalse),
        ("GeoMeanSCSApproxTrue", GeoMeanSCSApproxTrue),
        ("GeoMeanSCSApproxFalse", GeoMeanSCSApproxFalse),
    ]

    print("Running power cone benchmarks...")
    print("Comparing approx=True (SOC) vs approx=False (power cones)\n")
    print(f"{'Benchmark':<40} {'Compile (s)':<15} {'Solve (s)':<15}")
    print("=" * 70)

    for name, cls in benchmarks:
        try:
            bench = cls()
            bench.setup()

            # Time compilation
            start = time.time()
            bench.time_compile_problem()
            compile_time = time.time() - start

            # Time solve
            start = time.time()
            bench.time_solve_problem()
            solve_time = time.time() - start

            print(f"{name:<40} {compile_time:<15.4f} {solve_time:<15.4f}")
        except Exception as e:
            print(f"{name:<40} ERROR: {e}")

    print("\nDone!")
