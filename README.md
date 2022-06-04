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