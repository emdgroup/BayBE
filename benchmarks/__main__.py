"""Executes the benchmarking module."""
# Run this via 'python -m benchmarks' from the root directory.

import os
from multiprocessing import Pool

from benchmarks.domains import BENCHMARKS
from benchmarks.execution import run_benchmark


def main():
    """Run all benchmarks in parallel."""
    num_processes_to_spawn = min(os.cpu_count(), len(BENCHMARKS))
    with Pool(processes=num_processes_to_spawn) as pool:
        pool.map(run_benchmark, BENCHMARKS)


if __name__ == "__main__":
    main()
