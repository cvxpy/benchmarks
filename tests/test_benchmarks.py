from benchmark.benchmark_suite import CurrentVersionBenchmarkSuite

# Running all benchmarks as a functionality for now
# At a later stage, testing the benchmark framework itself
# should be separated from running all benchmarks


def test_run_benchmarks():
    suite = CurrentVersionBenchmarkSuite()
    suite.register_default_benchmarks()
    suite.run_benchmarks()
