# SPDX-License-Identifier: Apache-2.0
"""Flickr30K native data access."""

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

BENCHMARK_NAME = "flickr30k"
ALIASES = ("flickr30k", "flickr", "flickr-30k", "flickr_30k", "filickr30k", "filickr_30k")
DEFAULT_SOURCE = "lmms-lab/flickr30k"
DEFAULT_SPLIT = "test"
LOCAL_NAMES = ("flickr30k", "Flickr30K")


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
    refs = as_string_list(
        row.get("annotations_captions")
        or row.get("captions")
        or row.get("caption")
        or row.get("sentences")
        or row.get("references")
    )
    media = row.get("image") or row.get("image_path") or row.get("file_name") or row.get("filename")
    sample_id = str(row.get("image_id") or row.get("img_id") or row.get("id") or row.get("filename") or index)
    return [
        BenchmarkSample(
            benchmark=BENCHMARK_NAME,
            sample_id=sample_id,
            prompt=f"{IMAGE_PLACEHOLDER}\nDescribe the image in one concise sentence.",
            media=[media],
            media_sources=[media_source(media, row)],
            references=refs,
            task_type="caption",
            config=config,
            split=split,
            native=sanitize_native(row),
            max_new_tokens=64,
        )
    ]


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
