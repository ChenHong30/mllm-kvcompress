# SPDX-License-Identifier: Apache-2.0
"""MMMU native data access."""

from typing import Any

from mllm_kvcompress.benchmarks.adapters import IMAGE_PLACEHOLDER
from mllm_kvcompress.benchmarks.base import BenchmarkSpec
from mllm_kvcompress.benchmarks.data import (
    BenchmarkSample,
    as_string_list,
    format_choices,
    letter_to_index,
    load_samples_from_source,
    media_source,
    replace_numbered_media_tags,
    sanitize_native,
)
from mllm_kvcompress.benchmarks.eval import evaluate_predictions

BENCHMARK_NAME = "mmmu"
ALIASES = ("mmmu",)
DEFAULT_SOURCE = "MMMU/MMMU"
DEFAULT_SPLIT = "validation"
LOCAL_NAMES = ("MMMU",)
DEFAULT_CONFIGS = [
    "Accounting",
    "Agriculture",
    "Architecture_and_Engineering",
    "Art",
    "Art_Theory",
    "Basic_Medical_Science",
    "Biology",
    "Chemistry",
    "Clinical_Medicine",
    "Computer_Science",
    "Design",
    "Diagnostics_and_Laboratory_Medicine",
    "Economics",
    "Electronics",
    "Energy_and_Power",
    "Finance",
    "Geography",
    "History",
    "Literature",
    "Manage",
    "Marketing",
    "Materials",
    "Math",
    "Mechanical_Engineering",
    "Music",
    "Pharmacy",
    "Physics",
    "Psychology",
    "Public_Health",
    "Sociology",
]


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
    return [mmmu_sample(row, index, split, config)]


def mmmu_sample(row: dict[str, Any], index: int, split: str, config: str | None) -> BenchmarkSample:
    media = [row.get(f"image_{i}") for i in range(1, 8) if row.get(f"image_{i}") is not None]
    question = str(row.get("question", ""))
    prompt = replace_numbered_media_tags(question, len(media), IMAGE_PLACEHOLDER)
    if media and IMAGE_PLACEHOLDER not in prompt:
        prompt = "\n".join([IMAGE_PLACEHOLDER for _ in media] + [prompt])
    question_type = str(row.get("question_type", "")).lower()
    choices = as_string_list(row.get("options"))
    answer = str(row.get("answer", "")).strip()
    answer_index = letter_to_index(answer) if answer else None
    if choices and ("multiple" in question_type or len(answer) == 1):
        prompt = f"{prompt}\nOptions: {format_choices(choices)}\nJust output the letter of the correct answer."
        task_type = "multiple_choice"
        max_new_tokens = 8
    else:
        prompt = f"{prompt}\nAnswer the question directly."
        task_type = "open"
        max_new_tokens = 64
    return BenchmarkSample(
        benchmark=BENCHMARK_NAME,
        sample_id=str(row.get("id") or index),
        prompt=prompt,
        media=media,
        media_sources=[media_source(item, row) for item in media],
        references=[answer] if answer else [],
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
