# SPDX-License-Identifier: Apache-2.0
"""CLI runner for native benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from mllm_kvcompress import CompressionMethod
from mllm_kvcompress.benchmarks import BENCHMARKS, get_benchmark, load_model_adapter, normalize_benchmark_name, parse_benchmark_list
from mllm_kvcompress.benchmarks.data import (
    BenchmarkSample,
    load_sample_media,
)
from mllm_kvcompress.benchmarks.milebench.eval import write_csv_summary, write_json
from mllm_kvcompress.methods import parse_settings

ALL_STRATEGY_SETTINGS = [
    "baseline",
    "fastv_0.5",
    "fitprune_0.5",
    "gui_kv_0.5",
    "infinipot_v_0.5",
    "look_m_0.5",
    "meda_0.5",
    "mixkv_0.5",
    "sparsemm_0.5",
    "star_kv_0.5",
    "vidkv",
]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run mllm-kvcompress on native benchmarks.")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--benchmarks", default="all", help=f"Comma-separated benchmarks, or all: {list(BENCHMARKS)}")
    parser.add_argument(
        "--settings",
        default="all",
        help="Comma-separated compression settings. 'all' expands to baseline plus all implemented strategies at 50 percent.",
    )
    parser.add_argument("--output-dir", default="runs/native_benchmarks")
    parser.add_argument("--data-root", default=None, help="Optional root containing native dataset folders.")
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Override one benchmark data source as benchmark=path_or_hf_id. Can be repeated.",
    )
    parser.add_argument(
        "--benchmark-split",
        action="append",
        default=[],
        help="Override one benchmark split as benchmark=split. Can be repeated.",
    )
    parser.add_argument(
        "--benchmark-config",
        action="append",
        default=[],
        help="Override one benchmark config list as benchmark=config1,config2. Use all for MMMU/PCA defaults.",
    )
    parser.add_argument("--split", default=None, help="Global split override for all benchmarks.")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per benchmark/config for smoke runs.")
    parser.add_argument("--streaming", action="store_true", help="Use Hugging Face streaming datasets when possible.")
    parser.add_argument("--data-cache-dir", default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--check-data-only",
        action="store_true",
        help="Load samples and the first sample media for each benchmark, then exit without loading the model.",
    )
    parser.add_argument("--verbose-compression", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_native_benchmarks(args) -> dict[str, Any]:
    benchmarks = parse_benchmark_list(args.benchmarks)
    source_overrides = _parse_overrides(args.source)
    split_overrides = _parse_overrides(args.benchmark_split)
    config_overrides = _parse_overrides(args.benchmark_config)
    settings = parse_strategy_settings(args.settings)

    samples_by_benchmark: dict[str, list[BenchmarkSample]] = {}
    for benchmark in benchmarks:
        spec = get_benchmark(benchmark)
        source = source_overrides.get(benchmark) or spec.source_for(args.data_root)
        split = args.split or split_overrides.get(benchmark)
        config = config_overrides.get(benchmark)
        samples = spec.load_samples(
            source=source,
            split=split,
            config=config,
            data_root=args.data_root,
            limit=args.limit,
            streaming=args.streaming,
            cache_dir=args.data_cache_dir,
        )
        samples_by_benchmark[benchmark] = samples
        print(f"[data] {benchmark}: loaded {len(samples)} samples from {source}", flush=True)

    if args.check_data_only:
        return _check_data(args.output_dir, samples_by_benchmark)

    adapter = load_model_adapter(
        args.model,
        model_type="auto",
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )

    run_names = []
    for setting_name, method_factory in settings:
        run_model_name = f"{adapter.model_slug}__{setting_name}"
        run_names.append(run_model_name)
        for benchmark in benchmarks:
            output_dir = Path(args.output_dir) / run_model_name / benchmark
            pred_path = output_dir / "pred.json"
            if pred_path.exists() and not args.overwrite:
                print(f"[skip] {pred_path} exists; pass --overwrite to regenerate", flush=True)
                predictions = json.loads(pred_path.read_text(encoding="utf-8"))
            else:
                predictions = _generate_benchmark(
                    adapter,
                    samples_by_benchmark[benchmark],
                    setting_name,
                    method_factory,
                    args,
                    benchmark,
                )
                write_json(pred_path, predictions)

            if not args.skip_eval:
                metrics, eval_list = get_benchmark(benchmark).evaluate(predictions)
                write_json(output_dir / "eval.json", metrics, indent=None)
                write_json(output_dir / "eval_score.json", eval_list)
                print(f"[eval] {run_model_name}/{benchmark}: {metrics}", flush=True)

    if args.skip_eval:
        return {}
    return _summarize(args.output_dir, run_names, benchmarks)


def parse_strategy_settings(settings: str) -> list[tuple[str, Callable[[], CompressionMethod | None]]]:
    return parse_settings(settings, default=list(ALL_STRATEGY_SETTINGS))


def _generate_benchmark(
    adapter,
    samples: list[BenchmarkSample],
    setting_name: str,
    method_factory: Callable[[], CompressionMethod | None],
    args,
    benchmark: str,
):
    predictions = []
    start = time.time()
    for index, sample in enumerate(samples):
        sample_start = time.time()
        media = load_sample_media(sample)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens or sample.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "do_sample": args.do_sample,
        }
        if args.temperature > 0:
            gen_kwargs["temperature"] = args.temperature
        answer = adapter.generate(
            sample.prompt,
            media,
            method_factory=method_factory,
            gen_kwargs=gen_kwargs,
            verbose=args.verbose_compression,
        )
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "benchmark": sample.benchmark,
                "config": sample.config,
                "split": sample.split,
                "task_type": sample.task_type,
                "media": sample.media_sources,
                "question": sample.prompt,
                "references": sample.references,
                "choices": sample.choices,
                "answer": sample.answer,
                "answer_index": sample.answer_index,
                "gen_model_id": adapter.model_slug,
                "compression": setting_name,
                "pred_response": answer,
                "gen_kwargs": dict(gen_kwargs),
                "seconds": round(time.time() - sample_start, 4),
                "native": sample.native,
            }
        )
        if (index + 1) % 10 == 0 or (index + 1) == len(samples):
            seconds_per_sample = (time.time() - start) / (index + 1)
            print(
                f"[{setting_name}/{benchmark}] {index + 1}/{len(samples)} "
                f"({seconds_per_sample:.2f}s/sample)",
                flush=True,
            )
    return predictions


def _summarize(output_dir: str | Path, run_names: list[str], benchmarks: list[str]) -> dict[str, Any]:
    output_dir = Path(output_dir)
    result = {}
    for run_name in run_names:
        result[run_name] = {}
        for benchmark in benchmarks:
            eval_path = output_dir / run_name / benchmark / "eval.json"
            result[run_name][benchmark] = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    write_json(output_dir / "result.json", result)
    write_csv_summary(output_dir / "result.csv", {name: {"Native Benchmarks": metrics} for name, metrics in result.items()})
    return result


def _check_data(output_dir: str | Path, samples_by_benchmark: dict[str, list[BenchmarkSample]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for benchmark, samples in samples_by_benchmark.items():
        entry: dict[str, Any] = {"num_samples": len(samples), "first_sample": None, "first_media_load": None}
        if samples:
            sample = samples[0]
            entry["first_sample"] = {
                "sample_id": sample.sample_id,
                "task_type": sample.task_type,
                "config": sample.config,
                "split": sample.split,
                "num_media": len(sample.media),
                "num_references": len(sample.references),
                "num_choices": len(sample.choices or []),
                "answer_index": sample.answer_index,
                "media_sources": sample.media_sources,
            }
            try:
                media = load_sample_media(sample)
                entry["first_media_load"] = {
                    "ok": True,
                    "sizes": [list(item.size) for item in media],
                }
            except Exception as exception:
                entry["first_media_load"] = {"ok": False, "error": str(exception)}
        report[benchmark] = entry
        print(f"[check] {benchmark}: {entry}", flush=True)
    write_json(Path(output_dir) / "data_check.json", report)
    return report


def _parse_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        key, raw = value.split("=", 1)
        overrides[normalize_benchmark_name(key)] = raw
    return overrides


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    run_native_benchmarks(args)
    if args.streaming:
        # Some streamed HF media datasets can abort during Python finalization after
        # all outputs are written. Avoid running those third-party finalizers in CLI use.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
