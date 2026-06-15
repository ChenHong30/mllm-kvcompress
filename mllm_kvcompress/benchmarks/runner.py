# SPDX-License-Identifier: Apache-2.0
"""Public benchmark API."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Any

import torch

from mllm_kvcompress.methods import format_method_setting


def run_benchmarks(
    model: str,
    benchmarks: str | list[str] = "all",
    settings: str | list[str] = "baseline",
    output_dir: str | Path = "runs/native_benchmarks",
    data_root: str | Path | None = None,
    sources: dict[str, Any] | None = None,
    splits: dict[str, Any] | None = None,
    configs: dict[str, Any] | None = None,
    split: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
    data_cache_dir: str | None = None,
    device: str | None = None,
    dtype: str = "bf16",
    max_new_tokens: int | None = None,
    min_new_tokens: int = 1,
    temperature: float = 0.0,
    do_sample: bool = False,
    overwrite: bool = False,
    skip_eval: bool = False,
    check_data_only: bool = False,
    verbose_compression: bool = False,
    local_files_only: bool = False,
    trust_remote_code: bool = True,
) -> dict[str, Any]:
    """Run one or more compression settings on one or more registered benchmarks."""

    from mllm_kvcompress.benchmarks.native_runner import run_native_benchmarks

    args = Namespace(
        model=model,
        benchmarks=_format_list(benchmarks),
        settings=_format_list(settings),
        output_dir=str(output_dir),
        data_root=data_root,
        source=_format_overrides(sources or {}),
        benchmark_split=_format_overrides(splits or {}),
        benchmark_config=_format_overrides(configs or {}),
        split=split,
        limit=limit,
        streaming=streaming,
        data_cache_dir=data_cache_dir,
        device=device or ("cuda:0" if torch.cuda.is_available() else "cpu"),
        dtype=dtype,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        overwrite=overwrite,
        skip_eval=skip_eval,
        check_data_only=check_data_only,
        verbose_compression=verbose_compression,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return run_native_benchmarks(args)


def run_benchmark(
    model: str,
    benchmark: str,
    setting: str | None = None,
    method: str | None = None,
    method_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run one compression setting on one registered benchmark."""

    if setting is not None and method is not None:
        raise ValueError("Pass either setting or method, not both.")
    resolved_setting = setting or (format_method_setting(method, method_kwargs) if method else "baseline")
    return run_benchmarks(model=model, benchmarks=benchmark, settings=resolved_setting, **kwargs)


def _format_list(value: str | list[str]) -> str:
    return ",".join(value) if isinstance(value, list) else value


def _format_overrides(values: dict[str, Any]) -> list[str]:
    return [f"{key}={_format_override_value(value)}" for key, value in values.items()]


def _format_override_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    return str(value)
