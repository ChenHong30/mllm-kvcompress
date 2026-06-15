# SPDX-License-Identifier: Apache-2.0
"""OCR-VQA native data access."""

from typing import Any

from mllm_kvcompress.benchmarks.adapters import IMAGE_PLACEHOLDER
from mllm_kvcompress.benchmarks.base import BenchmarkSpec
from mllm_kvcompress.benchmarks.data import (
    BenchmarkSample,
    as_string_list,
    load_samples_from_source,
    media_source,
    sanitize_native,
)
from mllm_kvcompress.benchmarks.eval import evaluate_predictions

BENCHMARK_NAME = "ocr_vqa"
ALIASES = ("ocr_vqa", "ocr-vqa", "ocrvqa")
DEFAULT_SOURCE = "howard-hou/OCR-VQA"
DEFAULT_SPLIT = "test"
LOCAL_NAMES = ("OCR-VQA", "OCR_VQA", "OCRVQA", "ocr_vqa")


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
        cap_expanded_samples=True,
    )


def row_to_samples(row: dict[str, Any], index: int, split: str, config: str | None):
    media = row.get("image") or row.get("image_path")
    questions = as_string_list(row.get("questions") or row.get("question"))
    answers = as_string_list(row.get("answers") or row.get("answer"))
    if not questions:
        questions = [""]
    if len(answers) == 1 and len(questions) > 1:
        answers = answers * len(questions)

    samples = []
    for q_idx, question in enumerate(questions):
        answer = answers[q_idx] if q_idx < len(answers) else ""
        sample_id = f"{row.get('image_id') or row.get('id') or index}_{q_idx}"
        samples.append(
            BenchmarkSample(
                benchmark=BENCHMARK_NAME,
                sample_id=str(sample_id),
                prompt=f"{IMAGE_PLACEHOLDER}\nQuestion: {question}\nAnswer the question with a short phrase.",
                media=[media],
                media_sources=[media_source(media, row)],
                references=[answer] if answer else [],
                answer=answer,
                task_type="ocr_vqa",
                config=config,
                split=split,
                native=sanitize_native({**row, "question": question, "answer": answer}),
                max_new_tokens=32,
            )
        )
    return samples


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
