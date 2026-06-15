# SPDX-License-Identifier: Apache-2.0
"""CLI runner for MileBench."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import torch

from mllm_kvcompress.core import CompressionMethod
from mllm_kvcompress.methods import parse_setting, parse_settings
from mllm_kvcompress.benchmarks import load_model_adapter
from mllm_kvcompress.benchmarks.milebench.data import (
    build_samples,
    core_subset_for_predictions,
    load_core_annotation,
    parse_dataset_list,
)
from mllm_kvcompress.benchmarks.milebench.eval import MileBenchEvaluator, summarize_results, write_json
from mllm_kvcompress.benchmarks.milebench.prepare import MILEBENCH_REPO_ID, ensure_milebench_data


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run mllm-kvcompress on MileBench.")
    parser.add_argument(
        "--datasets",
        default=MILEBENCH_REPO_ID,
        help="MileBench data source: a Hugging Face dataset id or a local folder containing archives/extracted data.",
    )
    parser.add_argument("--subset", default="all", help="Comma-separated MileBench subsets/groups to evaluate, or all.")
    parser.add_argument(
        "--data-root",
        default="data/MileBench",
        help="Local extraction folder used when --datasets is a Hugging Face dataset id.",
    )
    parser.add_argument("--download-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-cache-dir", default=None, help="Optional Hugging Face cache dir for MileBench archives.")
    parser.add_argument("--force-data-download", action="store_true")
    parser.add_argument("--model", required=True, help="Model name or local model path.")
    parser.add_argument("--split", default="test", choices=["test", "adv"])
    parser.add_argument("--settings", default="baseline", help="Comma-separated compression settings.")
    parser.add_argument("--output-dir", default="runs/milebench")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--max-context-len", type=int, default=None)
    parser.add_argument("--tokens-per-image", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset for smoke runs.")
    parser.add_argument("--combine-image", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--verbose-compression", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def run_milebench(args) -> dict[str, Any]:
    subsets = parse_dataset_list(args.subset)
    data_dir = _prepare_data_source(
        args.datasets,
        subsets,
        args,
    )
    adapter = load_model_adapter(
        args.model,
        model_type="auto",
        device=args.device,
        dtype=args.dtype,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    settings = parse_settings(args.settings)
    max_context_len = args.max_context_len or adapter.default_max_context_len
    tokens_per_image = args.tokens_per_image or adapter.default_tokens_per_image
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    setting_names = []
    for setting_name, method_factory in settings:
        run_model_name = f"{adapter.model_slug}__{setting_name}"
        setting_names.append(run_model_name)
        for dataset_name in subsets:
            dataset_output_dir = Path(args.output_dir) / run_model_name / dataset_name
            pred_path = dataset_output_dir / "pred.json"
            core_annotation = load_core_annotation(
                data_dir, dataset_name, split=args.split, combine_image=args.combine_image
            )
            if pred_path.exists() and not args.overwrite:
                print(f"[skip] {pred_path} exists; pass --overwrite to regenerate", flush=True)
                predictions = json.loads(pred_path.read_text(encoding="utf-8"))
            else:
                samples = build_samples(
                    core_annotation,
                    data_dir,
                    dataset_name,
                    adapter,
                    max_context_len=max_context_len,
                    tokens_per_image=tokens_per_image,
                    combine_image=args.combine_image,
                    limit=args.limit,
                )
                predictions = _generate_dataset(
                    adapter,
                    samples,
                    setting_name,
                    method_factory,
                    gen_kwargs,
                    args.verbose_compression,
                    dataset_name,
                )
                write_json(pred_path, predictions)

            if not args.skip_eval:
                eval_core = core_subset_for_predictions(core_annotation, predictions)
                eval_result, eval_list, pred_with_extracted = MileBenchEvaluator().evaluate(
                    predictions, eval_core, dataset_name
                )
                write_json(dataset_output_dir / "eval.json", eval_result, indent=None)
                write_json(dataset_output_dir / "eval_score.json", eval_list)
                if pred_with_extracted is not None:
                    write_json(dataset_output_dir / "pred_with_extracted.json", pred_with_extracted)
                print(f"[eval] {run_model_name}/{dataset_name}: {eval_result}", flush=True)

    if not args.skip_eval:
        return summarize_results(args.output_dir, setting_names)
    return {}


def _prepare_data_source(data_source: str, subsets: list[str], args) -> Path:
    if _is_hf_repo_id(data_source):
        return ensure_milebench_data(
            args.data_root,
            subsets,
            split=args.split,
            combine_image=args.combine_image,
            download=args.download_data,
            repo_id=data_source,
            cache_dir=args.data_cache_dir,
            force_download=args.force_data_download,
        )

    return ensure_milebench_data(
        Path(data_source).expanduser(),
        subsets,
        split=args.split,
        combine_image=args.combine_image,
        download=args.download_data,
        repo_id=MILEBENCH_REPO_ID,
        cache_dir=args.data_cache_dir,
        force_download=args.force_data_download,
    )


def _is_hf_repo_id(value: str) -> bool:
    if Path(value).expanduser().exists():
        return False
    if value.startswith(("/", "./", "../", "~")):
        return False
    return re.fullmatch(r"[\w.-]+/[\w.-]+", value) is not None


def _generate_dataset(
    adapter,
    samples,
    setting_name: str,
    method_factory: Callable[[], CompressionMethod | None],
    gen_kwargs: dict[str, Any],
    verbose_compression: bool,
    dataset_name: str,
):
    predictions = []
    start = time.time()
    for index, sample in enumerate(samples):
        sample_start = time.time()
        images = adapter.load_images(sample.image_paths)
        answer = adapter.generate(
            sample.prompt,
            images,
            method_factory=method_factory,
            gen_kwargs=gen_kwargs,
            verbose=verbose_compression,
        )
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "image": sample.image_paths,
                "question": sample.prompt,
                "gt_response": sample.gt_response,
                "gen_model_id": adapter.model_slug,
                "compression": setting_name,
                "pred_response": answer,
                "gen_kwargs": dict(gen_kwargs),
                "seconds": round(time.time() - sample_start, 4),
            }
        )
        if (index + 1) % 10 == 0 or (index + 1) == len(samples):
            seconds_per_sample = (time.time() - start) / (index + 1)
            print(
                f"[{setting_name}/{dataset_name}] {index + 1}/{len(samples)} "
                f"({seconds_per_sample:.2f}s/sample)",
                flush=True,
            )
    return predictions


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    run_milebench(args)


if __name__ == "__main__":
    main()
