# SPDX-License-Identifier: Apache-2.0
"""PCA-Bench native data access."""

from typing import Any

from mllm_kvcompress.benchmarks.adapters import IMAGE_PLACEHOLDER
from mllm_kvcompress.benchmarks.base import BenchmarkSpec
from mllm_kvcompress.benchmarks.data import (
    BenchmarkSample,
    as_string_list,
    extract_media,
    first_present,
    format_choices,
    letter,
    load_samples_from_source,
    media_source,
    nonnegative_int,
    replace_numbered_media_tags,
    sanitize_native,
)
from mllm_kvcompress.benchmarks.eval import evaluate_predictions

BENCHMARK_NAME = "pca_bench"
ALIASES = ("pca_bench", "pca-bench", "pcabench", "pca")
DEFAULT_SOURCE = "PCA-Bench/PCA-Bench-V1"
DEFAULT_SPLIT = "test_closed"
LOCAL_NAMES = ("PCA-Bench", "PCA_Bench")
DEFAULT_CONFIGS = ["Autonomous Driving", "Domestic Robot", "Open-World Game"]


def load_samples(
    source: str | None = None,
    split: str | None = None,
    config: str | None = None,
    data_root=None,
    limit: int | None = None,
    streaming: bool = False,
    cache_dir: str | None = None,
):
    return load_samples_from_source(
        source or BENCHMARK.source_for(data_root),
        split or DEFAULT_SPLIT,
        parse_configs(config),
        row_converter=row_to_samples,
        limit=limit,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def parse_configs(config: str | None) -> list[str | None]:
    if config and config != "all":
        return [item.strip() for item in config.replace(";", ",").split(",") if item.strip()]
    return list(DEFAULT_CONFIGS)


def row_to_samples(row: dict[str, Any], index: int, split: str, config: str | None):
    return [pca_sample(row, index, split, config)]


def pca_sample(row: dict[str, Any], index: int, split: str, config: str | None) -> BenchmarkSample:
    media = extract_media(row)
    question = str(row.get("question") or row.get("prompt") or row.get("query") or row.get("instruction") or "")
    question = replace_numbered_media_tags(question, len(media), IMAGE_PLACEHOLDER)
    if media and IMAGE_PLACEHOLDER not in question:
        question = "\n".join([IMAGE_PLACEHOLDER for _ in media] + [question])
    choices = as_string_list(first_present(row, "choices", "options", "actions", "candidates"))
    answer_index = nonnegative_int(first_present(row, "answer_index", "label", "correct_index"))
    answer = str(first_present(row, "answer", "ground_truth", "target", default="")).strip()
    if answer_index is not None and not answer:
        answer = letter(answer_index)
    closed = bool(choices) or "closed" in split
    if closed and choices:
        prompt = f"{question}\nOptions: {format_choices(choices)}\nJust output the letter of the correct answer."
        task_type = "multiple_choice"
        max_new_tokens = 8
    else:
        prompt = f"{question}\nAnswer the question directly."
        task_type = "open"
        max_new_tokens = 96
    return BenchmarkSample(
        benchmark=BENCHMARK_NAME,
        sample_id=str(row.get("id") or row.get("question_id") or index),
        prompt=prompt,
        media=media,
        media_sources=[media_source(item, row) for item in media],
        references=[answer] if answer else as_string_list(row.get("answers")),
        choices=choices or None,
        answer=answer,
        answer_index=answer_index,
        task_type=task_type,
        config=config,
        split=split,
        native=sanitize_native(row),
        max_new_tokens=max_new_tokens,
    )


def evaluate(predictions):
    return evaluate_predictions(BENCHMARK_NAME, predictions)


BENCHMARK = BenchmarkSpec(
    name=BENCHMARK_NAME,
    aliases=ALIASES,
    default_split=DEFAULT_SPLIT,
    default_source=DEFAULT_SOURCE,
    load_samples=load_samples,
    evaluate=evaluate,
    local_names=LOCAL_NAMES,
)
