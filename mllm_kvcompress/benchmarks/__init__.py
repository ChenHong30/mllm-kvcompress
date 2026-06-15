# SPDX-License-Identifier: Apache-2.0
"""Benchmark runners and model adapters for mllm_kvcompress."""

from mllm_kvcompress.benchmarks.adapters import BaseModelAdapter, load_model_adapter
from mllm_kvcompress.benchmarks.aokvqa import BENCHMARK as AOKVQA
from mllm_kvcompress.benchmarks.base import BenchmarkSpec
from mllm_kvcompress.benchmarks.flickr30k import BENCHMARK as FLICKR30K
from mllm_kvcompress.benchmarks.mmmu import BENCHMARK as MMMU
from mllm_kvcompress.benchmarks.nocaps import BENCHMARK as NOCAPS
from mllm_kvcompress.benchmarks.ocr_vqa import BENCHMARK as OCR_VQA
from mllm_kvcompress.benchmarks.pca_bench import BENCHMARK as PCA_BENCH

BENCHMARKS = {
    "nocaps": NOCAPS,
    "flickr30k": FLICKR30K,
    "aokvqa": AOKVQA,
    "mmmu": MMMU,
    "pca_bench": PCA_BENCH,
    "ocr_vqa": OCR_VQA,
}

_BENCHMARK_ALIASES = {
    alias.strip().lower().replace(" ", "_"): benchmark.name
    for benchmark in BENCHMARKS.values()
    for alias in benchmark.aliases
}


def normalize_benchmark_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    if normalized not in _BENCHMARK_ALIASES:
        raise KeyError(f"Unknown benchmark '{name}', available: {sorted(BENCHMARKS)}")
    return _BENCHMARK_ALIASES[normalized]


def get_benchmark(name: str) -> BenchmarkSpec:
    return BENCHMARKS[normalize_benchmark_name(name)]


def create_benchmark(name: str) -> BenchmarkSpec:
    """Return a benchmark registry entry by name or alias."""

    return get_benchmark(name)


def parse_benchmark_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return list(BENCHMARKS)
    return [normalize_benchmark_name(item) for item in value.replace(";", ",").split(",") if item.strip()]


def run_benchmark(*args, **kwargs):
    from mllm_kvcompress.benchmarks.runner import run_benchmark as _run_benchmark

    return _run_benchmark(*args, **kwargs)


def run_benchmarks(*args, **kwargs):
    from mllm_kvcompress.benchmarks.runner import run_benchmarks as _run_benchmarks

    return _run_benchmarks(*args, **kwargs)


__all__ = [
    "AOKVQA",
    "BENCHMARKS",
    "BaseModelAdapter",
    "BenchmarkSpec",
    "FLICKR30K",
    "MMMU",
    "NOCAPS",
    "OCR_VQA",
    "PCA_BENCH",
    "create_benchmark",
    "get_benchmark",
    "load_model_adapter",
    "normalize_benchmark_name",
    "parse_benchmark_list",
    "run_benchmark",
    "run_benchmarks",
]
