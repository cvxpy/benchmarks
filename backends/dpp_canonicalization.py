"""
DPP (Disciplined Parameterized Programming) Canonicalization Benchmarks

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

This module provides comprehensive benchmarks for DPP canonicalization,
comparing arbitrary backends (SCIPY, COO, NUMPY, etc.) on:
1. Large parameter matrices (100K-1M parameters)
2. Cold path (first canonicalization) vs warm path (re-solve)
3. Various problem structures (LP, QP, LASSO, SVM, Portfolio)

NOT for CI - run times may be substantial for large configurations.

Usage (CLI):
    python benchmark/dpp_canonicalization.py [options]

    --quick: Run only small configurations
    --full: Run all benchmarks including huge problems
    --backends COO SCIPY: Specify backends to compare (default: SCIPY COO)
    --scaling: Run detailed scaling analysis
    --profile: Enable cProfile output for bottleneck analysis
    --dpp-only: Only run DPP problems (skip non-DPP)
    --ignore-dpp: Also run with ignore_dpp=True for comparison

ASV Usage:
    asv run --bench DPP
"""

import argparse
import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import cvxpy as cp
import numpy as np

# Optional imports
try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import cProfile
    import pstats
    from io import StringIO
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

try:
    import tracemalloc
    HAS_TRACEMALLOC = True
except ImportError:
    HAS_TRACEMALLOC = False


# =============================================================================
# Configuration (for ASV)
# =============================================================================

@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    n_vars: int
    n_constraints: int
    param_size: int
    problem_type: str  # 'LP', 'QP', 'SOCP'

    def __str__(self):
        return f"{self.name} ({self.param_size:,} params)"


QUICK_CONFIGS = [
    BenchmarkConfig("tiny_lp", 50, 50, 2500, "LP"),
    BenchmarkConfig("small_lp", 100, 100, 10000, "LP"),
    BenchmarkConfig("medium_lp", 100, 500, 50000, "LP"),
]

FULL_CONFIGS = [
    BenchmarkConfig("tiny_lp", 50, 50, 2500, "LP"),
    BenchmarkConfig("small_lp", 100, 100, 10000, "LP"),
    BenchmarkConfig("medium_lp", 100, 500, 50000, "LP"),
    BenchmarkConfig("large_lp", 200, 500, 100000, "LP"),
    BenchmarkConfig("xlarge_lp", 500, 500, 250000, "LP"),
    BenchmarkConfig("xxlarge_lp", 1000, 500, 500000, "LP"),
    BenchmarkConfig("huge_lp", 1000, 1000, 1000000, "LP"),
    BenchmarkConfig("small_qp", 100, 100, 10000, "QP"),
    BenchmarkConfig("medium_qp", 200, 200, 40000, "QP"),
    BenchmarkConfig("large_qp", 500, 200, 100000, "QP"),
]

SCALING_CONFIGS = [
    BenchmarkConfig(f"scale_{n}x{n}", n, n, n*n, "LP")
    for n in [50, 100, 150, 200, 300, 400, 500, 700, 1000]
]


# =============================================================================
# ASV Problem Generators
# =============================================================================

def create_param_lp(n_vars: int, n_constraints: int) -> tuple:
    """
    Create LP with parametrized constraint matrix: A_param @ x <= b
    Parameter count = n_constraints * n_vars
    """
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_param = cp.Parameter(n_constraints)
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A_param @ x <= b_param, x >= 0]
    )

    def init_params():
        A_param.value = np.random.randn(n_constraints, n_vars)
        b_param.value = np.random.randn(n_constraints) + 10

    return prob, init_params, n_constraints * n_vars


def create_param_qp(n_vars: int, n_constraints: int) -> tuple:
    """
    Create QP with parametrized quadratic term: x'Qx + c'x
    Uses diagonal Q for DPP compliance.
    """
    x = cp.Variable(n_vars)
    q_diag = cp.Parameter(n_vars, nonneg=True)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_param = cp.Parameter(n_constraints)
    c = np.random.randn(n_vars)

    objective = 0.5 * cp.sum(cp.multiply(q_diag, cp.square(x))) + c @ x

    prob = cp.Problem(
        cp.Minimize(objective),
        [A_param @ x <= b_param, x >= -10, x <= 10]
    )

    def init_params():
        q_diag.value = np.abs(np.random.randn(n_vars)) + 0.1
        A_param.value = np.random.randn(n_constraints, n_vars)
        b_param.value = np.random.randn(n_constraints) + 10

    param_count = n_vars + n_constraints * n_vars
    return prob, init_params, param_count


def create_param_sparse_lp(n_vars: int, n_constraints: int, density: float = 0.1) -> tuple:
    """Create LP with sparse parametrized constraint matrix."""
    x = cp.Variable(n_vars)
    A_param = cp.Parameter((n_constraints, n_vars))
    b_param = cp.Parameter(n_constraints)
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A_param @ x <= b_param, x >= 0]
    )

    def init_params():
        A_param.value = sp.random(n_constraints, n_vars, density=density).toarray()
        b_param.value = np.random.randn(n_constraints) + 10

    return prob, init_params, n_constraints * n_vars


def create_elementwise_param_lp(n_vars: int, n_constraints: int) -> tuple:
    """Create LP with element-wise parameter multiplication."""
    x = cp.Variable(n_vars)
    scale_param = cp.Parameter(n_vars, nonneg=True)
    A = np.random.randn(n_constraints, n_vars)
    b = np.random.randn(n_constraints) + 10
    c = np.random.randn(n_vars)

    prob = cp.Problem(
        cp.Minimize(c @ x),
        [A @ cp.multiply(scale_param, x) <= b, x >= 0]
    )

    def init_params():
        scale_param.value = np.abs(np.random.randn(n_vars)) + 0.1

    return prob, init_params, n_vars


# =============================================================================
# ASV Benchmark Classes
# =============================================================================

class DPPLargeLPColdPath:
    """Benchmark cold path canonicalization for large parametrized LP (100K params)."""
    timeout = 300

    def setup(self):
        np.random.seed(42)
        n_vars, n_constraints = 200, 500
        self.prob, self.init_params, self.param_count = create_param_lp(n_vars, n_constraints)
        self.init_params()

    def time_cold_path_scipy(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_cold_path_coo(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='COO')


class DPPMediumLPColdPath:
    """Benchmark cold path canonicalization for medium parametrized LP (50K params)."""

    def setup(self):
        np.random.seed(42)
        n_vars, n_constraints = 100, 500
        self.prob, self.init_params, self.param_count = create_param_lp(n_vars, n_constraints)
        self.init_params()

    def time_cold_path_scipy(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_cold_path_coo(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='COO')


class DPPSmallLPColdPath:
    """Benchmark cold path canonicalization for small parametrized LP (10K params)."""

    def setup(self):
        np.random.seed(42)
        n_vars, n_constraints = 100, 100
        self.prob, self.init_params, self.param_count = create_param_lp(n_vars, n_constraints)
        self.init_params()

    def time_cold_path_scipy(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_cold_path_coo(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='COO')


class DPPWarmPath:
    """Benchmark warm path (parameter update and re-application)."""

    def setup(self):
        np.random.seed(42)
        n_vars, n_constraints = 100, 500
        self.prob, self.init_params, self.param_count = create_param_lp(n_vars, n_constraints)
        self.init_params()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_warm_path_scipy(self):
        self.init_params()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_warm_path_coo(self):
        self.init_params()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='COO')


class DPPQPBenchmark:
    """Benchmark parametrized QP with diagonal quadratic term."""

    def setup(self):
        np.random.seed(42)
        n_vars, n_constraints = 200, 200
        self.prob, self.init_params, self.param_count = create_param_qp(n_vars, n_constraints)
        self.init_params()

    def time_cold_path_scipy(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='SCIPY')

    def time_cold_path_coo(self):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend='COO')


class DPPBackendComparison:
    """Compare different canon backends on the same problem."""
    params = ['SCIPY', 'COO']

    def setup(self, backend):
        np.random.seed(42)
        n_vars, n_constraints = 100, 200
        self.prob, self.init_params, self.param_count = create_param_lp(n_vars, n_constraints)
        self.init_params()
        self.backend = backend

    def time_cold_path(self, backend):
        self.prob._cache = type(self.prob._cache)()
        self.prob.get_problem_data(cp.CLARABEL, canon_backend=backend)


# =============================================================================
# CLI Benchmark Result
# =============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    backend: str
    is_dpp: bool
    times_ms: list = field(default_factory=list)
    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    error: str = ""


# =============================================================================
# CLI Core Timing Function
# =============================================================================

def time_canonicalization(
    problem_factory: Callable[[], tuple[cp.Problem, Callable | None]],
    backend: str,
    warmup: int = 1,
    iterations: int = 5,
    ignore_dpp: bool = False,
) -> BenchmarkResult:
    """
    Time the canonicalization phase of a problem.

    Args:
        problem_factory: Factory function that creates (Problem, param_initializer or None)
        backend: Backend name ('COO', 'SCIPY', 'NUMPY', etc.)
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        ignore_dpp: If True, pass ignore_dpp=True to get_problem_data

    Returns:
        BenchmarkResult with timing statistics
    """
    times = []
    is_dpp = False

    for _ in range(warmup):
        prob, init_params = problem_factory()
        if init_params is not None:
            is_dpp = True
            init_params()
        try:
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend, ignore_dpp=ignore_dpp)
        except Exception as e:
            return BenchmarkResult(name="", backend=backend, is_dpp=is_dpp, error=str(e))
        del prob
        gc.collect()

    for _ in range(iterations):
        prob, init_params = problem_factory()
        if init_params is not None:
            is_dpp = True
            init_params()
        gc.collect()

        prob._cache = type(prob._cache)()

        start = time.perf_counter()
        try:
            prob.get_problem_data(cp.CLARABEL, canon_backend=backend, ignore_dpp=ignore_dpp)
        except Exception as e:
            return BenchmarkResult(name="", backend=backend, is_dpp=is_dpp, error=str(e))
        end = time.perf_counter()

        times.append((end - start) * 1000)
        del prob
        gc.collect()

    return BenchmarkResult(
        name="",
        backend=backend,
        is_dpp=is_dpp,
        times_ms=times,
        mean_ms=statistics.mean(times),
        std_ms=statistics.stdev(times) if len(times) > 1 else 0,
        min_ms=min(times),
        max_ms=max(times),
    )


# =============================================================================
# CLI Non-DPP Problem Factories
# =============================================================================

def make_dense_qp(n: int) -> Callable:
    """Dense quadratic program with n variables."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        Q = np.random.randn(n, n)
        Q = Q @ Q.T
        c = np.random.randn(n)
        A = np.random.randn(n // 2, n)
        b = np.random.randn(n // 2)

        obj = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)
        constraints = [A @ x <= b, x >= -1, x <= 1]
        return cp.Problem(obj, constraints), None
    return factory


def make_sparse_lp(n: int, m: int, density: float = 0.01) -> Callable:
    """Sparse linear program with n variables and m constraints."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        c = np.random.randn(n)
        A = sp.random(m, n, density=density, format='csc', random_state=42)
        b = np.random.randn(m)

        obj = cp.Minimize(c @ x)
        constraints = [A @ x <= b]
        return cp.Problem(obj, constraints), None
    return factory


def make_lasso(n: int, m: int) -> Callable:
    """LASSO problem with n features and m samples."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        lambd = 0.1

        obj = cp.Minimize(0.5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1))
        return cp.Problem(obj), None
    return factory


def make_svm(n: int, m: int) -> Callable:
    """SVM problem with n features and m samples."""
    def factory():
        np.random.seed(42)
        w = cp.Variable(n)
        b = cp.Variable()
        xi = cp.Variable(m)

        X = np.random.randn(m, n)
        y = np.sign(np.random.randn(m))
        C = 1.0

        obj = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
        constraints = [
            cp.multiply(y, X @ w + b) >= 1 - xi,
            xi >= 0
        ]
        return cp.Problem(obj, constraints), None
    return factory


def make_portfolio(n: int) -> Callable:
    """Portfolio optimization with n assets."""
    def factory():
        np.random.seed(42)
        w = cp.Variable(n)
        mu = np.random.randn(n) * 0.1
        Sigma = np.random.randn(n, n)
        Sigma = Sigma @ Sigma.T / n
        gamma = 1.0

        obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
        constraints = [cp.sum(w) == 1, w >= 0]
        return cp.Problem(obj, constraints), None
    return factory


def make_sdp(n: int) -> Callable:
    """Semidefinite program with n x n matrix variable."""
    def factory():
        np.random.seed(42)
        X = cp.Variable((n, n), symmetric=True)
        C = np.random.randn(n, n)
        C = C + C.T
        A = np.random.randn(n, n)
        A = A + A.T

        obj = cp.Minimize(cp.trace(C @ X))
        constraints = [X >> 0, cp.trace(A @ X) == 1]
        return cp.Problem(obj, constraints), None
    return factory


def make_many_constraints(n_vars: int, n_constraints: int) -> Callable:
    """Problem with many small constraints."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(n_vars)
        constraints = []
        for i in range(n_constraints):
            np.random.seed(42 + i)
            a = np.random.randn(n_vars)
            constraints.append(a @ x <= np.random.randn())
        obj = cp.Minimize(cp.sum(x))
        return cp.Problem(obj, constraints), None
    return factory


def make_convolution(signal_len: int, kernel_len: int) -> Callable:
    """Problem involving convolution."""
    def factory():
        np.random.seed(42)
        x = cp.Variable(signal_len)
        kernel = np.random.randn(kernel_len)
        target = np.random.randn(signal_len + kernel_len - 1)

        obj = cp.Minimize(cp.sum_squares(cp.conv(kernel, x) - target))
        return cp.Problem(obj), None
    return factory


# =============================================================================
# CLI DPP Problem Factories
# =============================================================================

def make_dpp_dense_qp(n: int) -> Callable:
    """Dense QP with parametrized Q matrix."""
    def factory():
        x = cp.Variable(n)
        Q_param = cp.Parameter((n, n), PSD=True)
        c_param = cp.Parameter(n)
        A = np.random.randn(n // 2, n)
        b = np.random.randn(n // 2)

        obj = cp.Minimize(0.5 * cp.quad_form(x, Q_param) + c_param @ x)
        constraints = [A @ x <= b, x >= -1, x <= 1]
        prob = cp.Problem(obj, constraints)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            Q = np.random.randn(n, n)
            Q_param.value = Q @ Q.T
            c_param.value = np.random.randn(n)

        return prob, init_params
    return factory


def make_dpp_lasso(n: int, m: int) -> Callable:
    """LASSO with parametrized data matrix."""
    def factory():
        x = cp.Variable(n)
        A_param = cp.Parameter((m, n))
        b_param = cp.Parameter(m)
        lambd = 0.1

        obj = cp.Minimize(0.5 * cp.sum_squares(A_param @ x - b_param) + lambd * cp.norm(x, 1))
        prob = cp.Problem(obj)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A_param.value = np.random.randn(m, n)
            b_param.value = np.random.randn(m)

        return prob, init_params
    return factory


def make_dpp_svm(n: int, m: int) -> Callable:
    """SVM with parametrized data."""
    def factory():
        w = cp.Variable(n)
        b = cp.Variable()
        xi = cp.Variable(m)

        X_param = cp.Parameter((m, n))
        y_param = cp.Parameter(m)
        C = 1.0

        obj = cp.Minimize(0.5 * cp.sum_squares(w) + C * cp.sum(xi))
        constraints = [
            cp.multiply(y_param, X_param @ w + b) >= 1 - xi,
            xi >= 0
        ]
        prob = cp.Problem(obj, constraints)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            X_param.value = np.random.randn(m, n)
            y_param.value = np.sign(np.random.randn(m))

        return prob, init_params
    return factory


def make_dpp_portfolio(n: int) -> Callable:
    """Portfolio optimization with parametrized returns and covariance."""
    def factory():
        w = cp.Variable(n)
        mu_param = cp.Parameter(n)
        Sigma_param = cp.Parameter((n, n), PSD=True)
        gamma = 1.0

        obj = cp.Maximize(mu_param @ w - gamma * cp.quad_form(w, Sigma_param))
        constraints = [cp.sum(w) == 1, w >= 0]
        prob = cp.Problem(obj, constraints)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            mu_param.value = np.random.randn(n) * 0.1
            Sigma = np.random.randn(n, n)
            Sigma_param.value = Sigma @ Sigma.T / n

        return prob, init_params
    return factory


def make_dpp_constraint_matrix(n_vars: int, n_constraints: int) -> Callable:
    """LP with fully parametrized constraint matrix - canonical DPP problem."""
    def factory():
        x = cp.Variable(n_vars)
        A_param = cp.Parameter((n_constraints, n_vars))
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)

        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A_param.value = np.random.randn(n_constraints, n_vars)

        return prob, init_params
    return factory


def make_dpp_multi_param(n_vars: int, n_constraints: int) -> Callable:
    """LP with multiple parametrized matrices."""
    def factory():
        x = cp.Variable(n_vars)
        A1 = cp.Parameter((n_constraints // 2, n_vars))
        A2 = cp.Parameter((n_constraints // 2, n_vars))
        b1 = np.random.randn(n_constraints // 2) + 10
        b2 = np.random.randn(n_constraints // 2) + 10
        c = np.random.randn(n_vars)

        prob = cp.Problem(
            cp.Minimize(c @ x),
            [A1 @ x <= b1, A2 @ x <= b2, x >= 0]
        )

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A1.value = np.random.randn(n_constraints // 2, n_vars)
            A2.value = np.random.randn(n_constraints // 2, n_vars)

        return prob, init_params
    return factory


def make_dpp_elementwise(n_vars: int, n_constraints: int) -> Callable:
    """LP with element-wise parameter scaling."""
    def factory():
        x = cp.Variable(n_vars)
        d_param = cp.Parameter(n_constraints, nonneg=True)
        A = np.random.randn(n_constraints, n_vars)
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)

        prob = cp.Problem(
            cp.Minimize(c @ x),
            [cp.multiply(d_param.reshape((n_constraints, 1)), A @ x) <= b, x >= 0]
        )

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            d_param.value = np.abs(np.random.randn(n_constraints)) + 0.1

        return prob, init_params
    return factory


def make_dpp_many_constraints(n_vars: int, n_constraints: int) -> Callable:
    """Many small constraints with parametrized coefficients."""
    def factory():
        x = cp.Variable(n_vars)
        params = [cp.Parameter(n_vars) for _ in range(n_constraints)]
        constraints = [p @ x <= 1 for p in params]
        obj = cp.Minimize(cp.sum(x))
        prob = cp.Problem(obj, constraints)

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            for p in params:
                p.value = np.random.randn(n_vars)

        return prob, init_params
    return factory


def make_dpp_sparse_constraint(n_vars: int, n_constraints: int, density: float = 0.1) -> Callable:
    """Sparse LP with parametrized constraint matrix."""
    def factory():
        x = cp.Variable(n_vars)
        A_param = cp.Parameter((n_constraints, n_vars))
        b = np.random.randn(n_constraints) + 10
        c = np.random.randn(n_vars)

        prob = cp.Problem(cp.Minimize(c @ x), [A_param @ x <= b, x >= 0])

        def init_params():
            np.random.seed(int(time.time() * 1000) % 2**31)
            A = np.random.randn(n_constraints, n_vars)
            mask = np.random.rand(n_constraints, n_vars) > density
            A[mask] = 0
            A_param.value = A

        return prob, init_params
    return factory


# =============================================================================
# CLI Benchmark Runner
# =============================================================================

def run_benchmark(
    name: str,
    problem_factory: Callable,
    backends: list[str],
    iterations: int = 5,
    warmup: int = 1,
    ignore_dpp: bool = False,
) -> dict[str, BenchmarkResult]:
    """Run a benchmark for a given problem across backends."""
    suffix = " [ignore_dpp]" if ignore_dpp else ""
    print(f"\n  {name}{suffix}")

    results = {}
    for backend in backends:
        result = time_canonicalization(
            problem_factory, backend, warmup=warmup, iterations=iterations,
            ignore_dpp=ignore_dpp
        )
        result.name = name
        results[backend] = result

        if result.error:
            print(f"    {backend:8s}: ERROR - {result.error[:50]}")
        else:
            print(f"    {backend:8s}: {result.mean_ms:8.2f}ms (±{result.std_ms:5.2f})")

    # Calculate speedups
    if len(backends) >= 2:
        base = backends[0]
        if base in results and not results[base].error:
            for other in backends[1:]:
                if other in results and not results[other].error:
                    speedup = results[base].mean_ms / results[other].mean_ms
                    winner = other if speedup > 1 else base
                    print(f"    {other} vs {base}: {speedup:.2f}x ({winner} faster)")

    return results


def run_benchmark_suite(
    title: str,
    benchmarks: list[tuple[str, Callable, int]],
    backends: list[str],
    ignore_dpp: bool = False,
) -> dict[str, dict]:
    """Run a suite of benchmarks."""
    suffix = " [ignore_dpp=True]" if ignore_dpp else ""
    print(f"\n{'=' * 70}")
    print(f"{title}{suffix}")
    print("=" * 70)

    all_results = {}
    for name, factory, iters in benchmarks:
        key = f"{name}_ignore_dpp" if ignore_dpp else name
        all_results[key] = run_benchmark(
            name, factory, backends, iterations=iters, ignore_dpp=ignore_dpp
        )

    return all_results


def print_summary(all_results: dict, backends: list[str]):
    """Print summary statistics."""
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    if len(backends) < 2:
        print("\nNeed at least 2 backends for comparison.")
        return

    base = backends[0]
    wins = {b: 0 for b in backends}
    speedups = {b: [] for b in backends[1:]}
    dpp_speedups = {b: [] for b in backends[1:]}
    non_dpp_speedups = {b: [] for b in backends[1:]}

    for name, results in all_results.items():
        if base not in results or results[base].error:
            continue

        base_time = results[base].mean_ms
        is_dpp = results[base].is_dpp

        fastest = base
        fastest_time = base_time

        for other in backends[1:]:
            if other not in results or results[other].error:
                continue

            other_time = results[other].mean_ms
            speedup = base_time / other_time
            speedups[other].append(speedup)

            if is_dpp:
                dpp_speedups[other].append(speedup)
            else:
                non_dpp_speedups[other].append(speedup)

            if other_time < fastest_time:
                fastest = other
                fastest_time = other_time

        wins[fastest] += 1

    total = sum(wins.values())
    print(f"\nWins out of {total} benchmarks:")
    for b in backends:
        print(f"  {b}: {wins[b]}")

    for other in backends[1:]:
        if speedups[other]:
            print(f"\n{other} vs {base}:")
            print(f"  Average speedup: {statistics.mean(speedups[other]):.2f}x")
            print(f"  Min: {min(speedups[other]):.2f}x, Max: {max(speedups[other]):.2f}x")

            if dpp_speedups[other]:
                print(f"  DPP problems: avg {statistics.mean(dpp_speedups[other]):.2f}x")
            if non_dpp_speedups[other]:
                print(f"  Non-DPP problems: avg {statistics.mean(non_dpp_speedups[other]):.2f}x")


def run_scaling_analysis(backends: list[str]):
    """Run detailed scaling analysis for DPP problems."""
    print(f"\n{'=' * 70}")
    print("SCALING ANALYSIS: Param Matrix Size vs Time")
    print("=" * 70)
    print("\nTesting A_param @ x <= b with varying A dimensions")
    print()

    sizes = [(50, 50), (100, 100), (200, 200), (300, 300), (500, 500),
             (200, 500), (500, 200), (1000, 500), (500, 1000)]

    header = f"{'Config':<12} {'Params':>10}"
    for b in backends:
        header += f" {b:>10}"
    if len(backends) >= 2:
        header += f" {'Speedup':>10}"
    print(header)
    print("-" * (35 + 12 * len(backends)))

    for m, k in sizes:
        factory = make_dpp_constraint_matrix(k, m)
        results = {}

        for backend in backends:
            result = time_canonicalization(factory, backend, warmup=1, iterations=3)
            results[backend] = result

        line = f"{m}x{k:<6} {m*k:>10,}"
        all_ok = True
        for b in backends:
            if results[b].error:
                line += f" {'ERROR':>10}"
                all_ok = False
            else:
                line += f" {results[b].mean_ms:>10.1f}"

        if len(backends) >= 2 and all_ok:
            speedup = results[backends[0]].mean_ms / results[backends[1]].mean_ms
            line += f" {speedup:>9.2f}x"

        print(line)


# =============================================================================
# Profiling
# =============================================================================

def run_detailed_profile(name: str, factory: Callable, backend: str):
    """Run detailed profiling for a specific problem."""
    if not HAS_PROFILER:
        print("cProfile not available")
        return

    print(f"\n{'=' * 70}")
    print(f"PROFILE: {name} with {backend}")
    print("=" * 70)

    prob, init_params = factory()
    if init_params:
        init_params()

    prob._cache = type(prob._cache)()

    pr = cProfile.Profile()
    pr.enable()
    prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())


# =============================================================================
# ASV-style run_benchmark (for compatibility)
# =============================================================================

def run_asv_benchmark(config: BenchmarkConfig,
                      backend: str = 'SCIPY',
                      n_warmups: int = 1,
                      n_runs: int = 3,
                      profile: bool = False):
    """Run a benchmark using ASV-style config."""
    np.random.seed(42)

    if config.problem_type == 'LP':
        prob, init_params, _ = create_param_lp(config.n_vars, config.n_constraints)
    elif config.problem_type == 'QP':
        prob, init_params, _ = create_param_qp(config.n_vars, config.n_constraints)
    else:
        raise ValueError(f"Unknown problem type: {config.problem_type}")

    init_params()

    peak_memory = 0.0
    if HAS_TRACEMALLOC:
        tracemalloc.start()

    cold_times = []
    for i in range(n_warmups + n_runs):
        prob._cache = type(prob._cache)()
        init_params()

        start = time.perf_counter()
        prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        elapsed = (time.perf_counter() - start) * 1000

        if i >= n_warmups:
            cold_times.append(elapsed)

    cold_time = np.median(cold_times)

    warm_times = []
    for _ in range(n_runs):
        init_params()
        start = time.perf_counter()
        prob.get_problem_data(cp.CLARABEL, canon_backend=backend)
        elapsed = (time.perf_counter() - start) * 1000
        warm_times.append(elapsed)

    warm_time = np.median(warm_times)

    if HAS_TRACEMALLOC:
        current, peak = tracemalloc.get_traced_memory()
        peak_memory = peak / (1024 * 1024)
        tracemalloc.stop()

    return {
        'config': config,
        'cold_time_ms': cold_time,
        'warm_time_ms': warm_time,
        'peak_memory_mb': peak_memory,
    }


def run_asv_scaling_analysis(configs: list[BenchmarkConfig], backend: str = 'SCIPY'):
    """Run scaling analysis with ASV-style configs."""
    print("=" * 80)
    print(f"DPP Canonicalization Scaling Analysis (Backend: {backend})")
    print("=" * 80)
    print()

    print(f"{'Config':<20} {'Params':>12} {'Cold (ms)':>12} {'Warm (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 70)

    results = []
    for config in configs:
        try:
            result = run_asv_benchmark(config, backend=backend)
            results.append(result)
            print(f"{config.name:<20} {config.param_size:>12,} {result['cold_time_ms']:>12.1f} "
                  f"{result['warm_time_ms']:>12.1f} {result['peak_memory_mb']:>12.1f}")
        except Exception as e:
            print(f"{config.name:<20} {config.param_size:>12,} ERROR: {e}")

    print()

    if len(results) >= 3:
        params = np.array([r['config'].param_size for r in results])
        times = np.array([r['cold_time_ms'] for r in results])

        b, log_a = np.polyfit(np.log(params), np.log(times), 1)

        print(f"Scaling: time ~ params^{b:.2f}")
        if b < 1.2:
            print("  -> Near-linear scaling (good)")
        elif b < 1.5:
            print("  -> Slightly superlinear scaling")
        else:
            print("  -> Superlinear scaling (potential bottleneck)")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='DPP Canonicalization Benchmark Suite'
    )
    parser.add_argument('--quick', action='store_true',
                        help='Quick benchmarks only (small problems)')
    parser.add_argument('--full', action='store_true',
                        help='Full benchmark suite including huge problems')
    parser.add_argument('--backends', nargs='+', default=['SCIPY', 'COO'],
                        help='Backends to compare (default: SCIPY COO)')
    parser.add_argument('--dpp-only', action='store_true',
                        help='Only run DPP (parametrized) benchmarks')
    parser.add_argument('--non-dpp-only', action='store_true',
                        help='Only run non-DPP (static) benchmarks')
    parser.add_argument('--ignore-dpp', action='store_true',
                        help='Also benchmark with ignore_dpp=True')
    parser.add_argument('--scaling', action='store_true',
                        help='Run detailed scaling analysis')
    parser.add_argument('--asv-scaling', action='store_true',
                        help='Run ASV-style scaling analysis')
    parser.add_argument('--profile', type=str, default=None,
                        help='Profile a specific problem (e.g., "dpp_lasso_medium")')
    args = parser.parse_args()

    backends = args.backends

    print("=" * 70)
    print("CVXPY Canonicalization Backend Benchmark Suite")
    print("=" * 70)
    print()
    print(f"Backends: {', '.join(backends)}")
    print()

    all_results = {}

    # Non-DPP benchmarks
    if not args.dpp_only:
        non_dpp_small = [
            ("Dense QP (n=50)", make_dense_qp(50), 5),
            ("Sparse LP (100x50)", make_sparse_lp(100, 50), 5),
            ("LASSO (50x100)", make_lasso(50, 100), 5),
            ("SVM (50x100)", make_svm(50, 100), 5),
            ("Portfolio (n=50)", make_portfolio(50), 5),
        ]

        non_dpp_medium = [
            ("Dense QP (n=200)", make_dense_qp(200), 3),
            ("Sparse LP (1000x500)", make_sparse_lp(1000, 500), 3),
            ("LASSO (200x500)", make_lasso(200, 500), 3),
            ("SVM (100x500)", make_svm(100, 500), 3),
            ("Portfolio (n=100)", make_portfolio(100), 3),
            ("SDP (n=30)", make_sdp(30), 3),
            ("Many constraints (50x500)", make_many_constraints(50, 500), 3),
            ("Convolution (1000x50)", make_convolution(1000, 50), 3),
        ]

        non_dpp_large = [
            ("Dense QP (n=500)", make_dense_qp(500), 3),
            ("Sparse LP (5000x2000)", make_sparse_lp(5000, 2000), 3),
            ("LASSO (500x2000)", make_lasso(500, 2000), 3),
        ]

        if args.quick:
            non_dpp_suite = non_dpp_small
        elif args.full:
            non_dpp_suite = non_dpp_small + non_dpp_medium + non_dpp_large
        else:
            non_dpp_suite = non_dpp_small + non_dpp_medium

        all_results.update(run_benchmark_suite(
            "NON-DPP PROBLEMS (Static)",
            non_dpp_suite,
            backends
        ))

    # DPP benchmarks
    if not args.non_dpp_only:
        dpp_small = [
            ("DPP Dense QP (n=50)", make_dpp_dense_qp(50), 5),
            ("DPP LASSO (50x100)", make_dpp_lasso(50, 100), 5),
            ("DPP SVM (50x100)", make_dpp_svm(50, 100), 5),
            ("DPP Portfolio (n=50)", make_dpp_portfolio(50), 5),
            ("DPP Constraint (100x100)", make_dpp_constraint_matrix(100, 100), 5),
        ]

        dpp_medium = [
            ("DPP Dense QP (n=200)", make_dpp_dense_qp(200), 3),
            ("DPP LASSO (200x500)", make_dpp_lasso(200, 500), 3),
            ("DPP SVM (100x500)", make_dpp_svm(100, 500), 3),
            ("DPP Portfolio (n=100)", make_dpp_portfolio(100), 3),
            ("DPP Constraint (500x200)", make_dpp_constraint_matrix(200, 500), 3),
            ("DPP Multi-param (200x500)", make_dpp_multi_param(200, 500), 3),
            ("DPP Elementwise (100x200)", make_dpp_elementwise(100, 200), 3),
            ("DPP Many constr (50x200)", make_dpp_many_constraints(50, 200), 3),
        ]

        dpp_large = [
            ("DPP Dense QP (n=500)", make_dpp_dense_qp(500), 3),
            ("DPP LASSO (500x2000)", make_dpp_lasso(500, 2000), 3),
            ("DPP Constraint (1000x500)", make_dpp_constraint_matrix(500, 1000), 3),
            ("DPP Constraint (500x1000)", make_dpp_constraint_matrix(1000, 500), 3),
            ("DPP Many constr (50x500)", make_dpp_many_constraints(50, 500), 3),
            ("DPP Sparse (1000x500)", make_dpp_sparse_constraint(500, 1000, 0.1), 3),
        ]

        dpp_huge = [
            ("DPP Constraint (2000x1000)", make_dpp_constraint_matrix(1000, 2000), 2),
            ("DPP Constraint (1000x2000)", make_dpp_constraint_matrix(2000, 1000), 2),
            ("DPP LASSO (1000x5000)", make_dpp_lasso(1000, 5000), 2),
        ]

        if args.quick:
            dpp_suite = dpp_small
        elif args.full:
            dpp_suite = dpp_small + dpp_medium + dpp_large + dpp_huge
        else:
            dpp_suite = dpp_small + dpp_medium + dpp_large

        all_results.update(run_benchmark_suite(
            "DPP PROBLEMS (Parametrized)",
            dpp_suite,
            backends
        ))

        if args.ignore_dpp:
            all_results.update(run_benchmark_suite(
                "DPP PROBLEMS with ignore_dpp=True",
                dpp_suite,
                backends,
                ignore_dpp=True
            ))

    # Scaling analysis
    if args.scaling:
        run_scaling_analysis(backends)

    # ASV-style scaling
    if args.asv_scaling:
        for backend in backends:
            run_asv_scaling_analysis(SCALING_CONFIGS, backend=backend)

    # Profiling
    if args.profile:
        profile_problems = {
            "dpp_lasso_small": ("DPP LASSO (50x100)", make_dpp_lasso(50, 100)),
            "dpp_lasso_medium": ("DPP LASSO (200x500)", make_dpp_lasso(200, 500)),
            "dpp_constraint_medium": (
                "DPP Constraint (500x200)", make_dpp_constraint_matrix(200, 500)
            ),
            "dpp_constraint_large": (
                "DPP Constraint (1000x500)", make_dpp_constraint_matrix(500, 1000)
            ),
        }
        if args.profile in profile_problems:
            name, factory = profile_problems[args.profile]
            for backend in backends:
                run_detailed_profile(name, factory, backend)
        else:
            print(f"Unknown profile target: {args.profile}")
            print(f"Available: {list(profile_problems.keys())}")

    # Summary
    print_summary(all_results, backends)


if __name__ == "__main__":
    main()
