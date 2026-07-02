# CVXPY benchmarks
[![Build Status](http://github.com/cvxpy/benchmarks/workflows/build/badge.svg?event=push)](https://github.com/cvxpy/benchmarks/actions/workflows/build.yml)  
Code and data related to CVXPY benchmarks.

## Motivation
Performance is an important consideration for the development of CVXPY.
The goal of this repo is to collect real-world examples of cvxpy problems and 
to make it easy to generate standardized benchmark reports.

## Getting started
To start benchmarking the CVXPY with Airspeed Velocity, clone this repository on your local system and install the dependencies.
Then run the benchmarks against the latest commit on CVXPY by:
```
asv run
```
To run a particular benchmark use:
```
asv run --bench CVaRBenchmark
```
## Canonicalization backend benchmarks
`benchmark/canonicalization_backends.py` compares CVXPY's canonicalization
backends on affine atom families, deep/wide expression trees, ND arrays,
broadcasting, einsum, convolve, kron, and dense-constant density regimes:
```
asv run --bench BackendCompileCanonicalization    # end-to-end get_problem_data
asv run --bench BackendBuildMatrixCanonicalization  # backend build_matrix only
asv run --bench DeepExpressionTreeScaling         # -(-(...-(x))), depth 4..256
asv run --bench WideExpressionTreeScaling         # sum of n matmul terms, n 8..256
```
All four are parameterized over `SCIPY`, `COO`, `CPP`, and `RUST`. Unavailable
backends (e.g. `RUST` when the `cvxpy_rust` extension is not installed, or
`CPP` on expressions the C++ core does not support) are reported as
unsupported for that parameter combination rather than failing. The module's
`LINOP_COVERAGE` dict maps every LinOp node type the backends process to the
cases exercising it.

To benchmark a locally built CVXPY (required for `RUST`, whose extension is
built by maturin/setuptools-rust rather than pip-installed from PyPI), run in
the current environment instead of an asv-managed one:
```
asv run --python=same --bench BackendCompileCanonicalization
```

You can also collect and view the results on a viewable website:
```
asv run
asv publish
asv preview
```

## Adding a benchmark
To create and test your own benchmark, you need to perform the following steps:
- Create a python file in the folder `benchmark/`.
- Create a class in this file, with the class name being the name of your benchmark. This class has to define 2 functions:
    - `setup()`
    - `time_compile_problem()`
- Add the definition of the CVXPY problem to the `setup()` function of the class.
- The `time_compile_problem()` function should only contain the call to `get_problem_data`.
- Run and test the benchmark inside the local environment on your system. Ideally, the benchmark should take between 5 and 10 seconds.
- Merge and check if it is correctly picked up by `cvxpy/cvxpy` repository and displayed on the dashboard.
