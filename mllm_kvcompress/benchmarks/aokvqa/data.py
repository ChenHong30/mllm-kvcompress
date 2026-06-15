# SPDX-License-Identifier: Apache-2.0
"""A-OKVQA native data access."""

from typing import Any

from mllm_kvcompress.benchmarks.adapters import IMAGE_PLACEHOLDER
from mllm_kvcompress.benchmarks.base import BenchmarkSpec
from mllm_kvcompress.benchmarks.data import (
    BenchmarkSample,
    as_string_list,
    format_choices,
    letter,
    load_samples_from_source,
    media_source,
    safe_int,
    sanitize_native,
)
from mllm_kvcompress.benchmarks.eval import evaluate_predictions

BENCHMARK_NAME = "aokvqa"
ALIASES = ("aokvqa", "a-okvqa", "a_okvqa")
DEFAULT_SOURCE = "HuggingFaceM4/A-OKVQA"
DEFAULT_SPLIT = "validation"
LOCAL_NAMES = ("A-OKVQA", "AOKVQA", "aokvqa")


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
        [config] if config else [None],
        row_converter=row_to_samples,
        limit=limit,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def row_to_samples(row: dict[str, Any], index: int, split: str, config: str | None):
    return [aokvqa_sample(row, index, split, config)]


def aokvqa_sample(row: dict[str, Any], index: int, split: str, config: str | None) -> BenchmarkSample:
    choices = as_string_list(row.get("choices"))
    answer_index = safe_int(row.get("correct_choice_idx"))
    answer = letter(answer_index) if answer_index is not None else str(row.get("answer", ""))
    question = str(row.get("question", ""))
    prompt = (
        f"{IMAGE_PLACEHOLDER}\nAnalyse the image and choose the best answer for the following question:\n"
        f"{question}\nOptions: {format_choices(choices)}\nJust output the letter of the correct answer."
    )
    media = row.get("image")
    return BenchmarkSample(
        benchmark=BENCHMARK_NAME,
        sample_id=str(row.get("question_id") or row.get("id") or index),
        prompt=prompt,
        media=[media],
        media_sources=[media_source(media, row)],
        references=[answer],
        choices=choices,
        answer=answer,
        answer_index=answer_index,
        task_type="multiple_choice",
        config=config,
        split=split,
        native=sanitize_native(row),
        max_new_tokens=8,
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
