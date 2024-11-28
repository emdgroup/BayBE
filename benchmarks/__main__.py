"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import os
from multiprocessing import Pool

from benchmarks.domains import BENCHMARKS, Benchmark
from benchmarks.persistence import (
    LocalFileObjectStorage,
    PathConstructor,
    S3ObjectStorage,
)

RUNS_IN_CI = "CI" in os.environ


def run_benchmark(benchmark: Benchmark) -> None:
    """Run a single benchmark and persist its result."""
    result = benchmark()
    path_constructor = PathConstructor.from_result(result)
    persist_dict = benchmark.to_dict() | result.to_dict()

    object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
    object_storage.write_json(persist_dict, path_constructor)


def main():
    """Run all benchmarks in parallel."""
    num_processes_to_spawn = min(os.process_cpu_count(), len(BENCHMARKS))
    with Pool(processes=num_processes_to_spawn) as pool:
        pool.map(run_benchmark, BENCHMARKS)


if __name__ == "__main__":
    main()
