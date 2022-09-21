# CVXPY benchmarks
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
asv run --bench cvar_benchmark.CVaRBenchmark
```
You can also collect and view the results on a viewable website:
```
asv run
asv publish
asv preview
```

## Writing a benchmark
To create and test your own benchmark, you need to follow the following steps:
- Create a python file in the folder `benchmark/`.
- Create a class in this file which should be the name of your benchmark. This class would define 2 functions:
    - `setup()`
    - `time_compile_problem()`
- If initialization needs to be performed that should not be included in the timing of the benchmark, include that code in a `setup()` method on the class.
- The problem compilation statement should be written inside the `time_compile_problem()` method.
- Run and test the benchmark inside the local environment on your system.
- Merge and check if it is correctly picked up by `cvxpy/cvxpy` repository and displayed in dashboard.
