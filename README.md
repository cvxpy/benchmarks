# CVXPY benchmarks
Code and data related to CVXPY benchmarks.

## Motivation
Performance is an important consideration for the development of CVXPY.
The goal of this repo is to collect real-world examples of cvxpy problems and 
to make it easy to generate standardized benchmark reports.

## Getting started
Running the current version of CVXPY on the default benchmarks is as simple as:
```python
from benchmark.benchmark_suite import CurrentVersionBenchmarkSuite

suite = CurrentVersionBenchmarkSuite()
suite.register_default_benchmarks()
suite.run_benchmarks()
```

## Interactive comparison
To compare two versions of CVXPY, you can use the interactive benchmarking mode:
```python
from benchmark.benchmark_suite import InteractiveComparisonBenchmarkSuite

suite = InteractiveComparisonBenchmarkSuite()
suite.register_default_benchmarks()
suite.run_benchmarks()

>? # checkout/install the first version of CVXPY to compare and provide a name
# e.g. run "pip install cvxpy==1.1.19" in a different terminal and type "1.1.19" in the input field
# Benchmarks are run

>? # checkout/install the second version of CVXPY to compare and provide a name
# Benchmarks are run again
# e.g. run "pip install cvxpy==1.2.0" in a different terminal and type "1.2.0" in the input field
```


## Benchmarking a custom problem
It is easily possible to create a benchmark report for a custom problem.
A minimal example is given below:
```python
import cvxpy as cp

from benchmark.benchmark_base import Benchmark
from benchmark.benchmark_suite import CurrentVersionBenchmarkSuite

class CustomProblem(Benchmark):
    @staticmethod
    def name() -> str:
        return "Custom Problem"

    @staticmethod
    def get_problem_instance() -> cp.Problem:
        # Problem generating code
        return problem

# Use any BenchmarkSuite to run the benchmark
benchmark_suite = CurrentVersionBenchmarkSuite()
benchmark_suite.register_benchmark(CustomProblem())
benchmark_suite.run_benchmarks()
```
