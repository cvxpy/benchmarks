import cvxpy as cp
import numpy as np

from benchmark.benchmark_base import Benchmark
from benchmark.benchmark_suite import CurrentVersionBenchmarkSuite


class CustomProblem(Benchmark):
    @staticmethod
    def name() -> str:
        return "Custom Problem"

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        # Problem data.
        m = 30
        n = 20
        np.random.seed(1)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        # Construct the problem.
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A @ x - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)
        return prob


def test_custom_problem():
    benchmark_suite = CurrentVersionBenchmarkSuite(output_file="")
    benchmark_suite.register_benchmark(CustomProblem())
    benchmark_suite.run_benchmarks()
