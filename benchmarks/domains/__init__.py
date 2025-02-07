"""Benchmark domains."""

from benchmarks.definition.config import Benchmark
from benchmarks.domains.synthetic_2C1D_1C import synthetic_2C1D_1C_benchmark
from benchmarks.domains.Hardness import hardness_benchmark, hardness_transfer_learning_benchmark

BENCHMARKS: list[Benchmark] = [
    synthetic_2C1D_1C_benchmark,
    hardness_benchmark,
    hardness_transfer_learning_benchmark, 
]

__all__ = ["BENCHMARKS"]
