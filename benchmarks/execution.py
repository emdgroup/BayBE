"""A separated module to execute benchmarks in parallel (https://github.com/python/cpython/issues/69240)."""

import os

from benchmarks.definition import Benchmark
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
    best_possible_value = benchmark.best_possible_value()
    dashboard_relevant_information = {
        "best_possible_result": best_possible_value
    } | result.to_dict()

    object_storage = S3ObjectStorage() if RUNS_IN_CI else LocalFileObjectStorage()
    object_storage.write_json(dashboard_relevant_information, path_constructor)
