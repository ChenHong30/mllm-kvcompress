# SPDX-License-Identifier: Apache-2.0
"""MileBench data loading and official-style prompt construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mllm_kvcompress.benchmarks.adapters import BaseModelAdapter, IMAGE_PLACEHOLDER

DATASET_GROUPS = {
    "Realistic Temporal": [
        "ActionLocalization",
        "ActionPrediction",
        "ActionSequence",
        "CharacterOrder",
        "CounterfactualInference",
        "EgocentricNavigation",
        "MovingAttribute",
        "MovingDirection",
        "ObjectExistence",
        "ObjectInteraction",
        "ObjectShuffle",
        "SceneTransition",
        "StateChange",
    ],
    "Realistic Semantic": [
        "ALFRED",
        "CLEVR-Change",
        "DocVQA",
        "IEdit",
        "MMCoQA",
        "MultiModalQA",
        "nuscenes",
        "OCR-VQA",
        "SlideVQA",
        "Spot-the-Diff",
        "TQA",
        "WebQA",
        "WikiVQA",
    ],
    "Diagnostic": ["TextNeedleInAHaystack", "ImageNeedleInAHaystack", "GPR1200"],
}

ALL_DATASETS = [dataset for datasets in DATASET_GROUPS.values() for dataset in datasets]

DATASET_ALIASES = {
    "CLEVR_Change": "CLEVR-Change",
    "OCR_VQA": "OCR-VQA",
    "Spot_the_Diff": "Spot-the-Diff",
}


@dataclass
class MileBenchSample:
    sample_id: int
    prompt: str
    image_paths: list[str]
    gt_response: str
    raw: dict[str, Any]


def normalize_dataset_name(name: str) -> str:
    name = name.strip()
    return DATASET_ALIASES.get(name, name)


def parse_dataset_list(value: str | None) -> list[str]:
    if not value or value == "all":
        return list(ALL_DATASETS)
    names = []
    for item in value.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if item in DATASET_GROUPS:
            names.extend(DATASET_GROUPS[item])
        else:
            names.append(normalize_dataset_name(item))
    unknown = sorted(set(names) - set(ALL_DATASETS))
    if unknown:
        raise ValueError(f"Unknown MileBench datasets: {unknown}. Available: {ALL_DATASETS}")
    return names


def load_core_annotation(
    data_dir: str | Path,
    dataset_name: str,
    split: str = "test",
    combine_image: int | None = None,
) -> dict[str, Any]:
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_dir = Path(data_dir) / dataset_name
    if split == "adv":
        filename = f"{dataset_name}-adv.json"
    elif combine_image and combine_image != 1:
        filename = f"{dataset_name}_combined_{combine_image}.json"
    else:
        filename = f"{dataset_name}.json"
    path = dataset_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MileBench annotation file: {path}. Download and extract FreedomIntelligence/MileBench first."
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def image_quantity_level(num_images: int) -> str:
    if num_images < 6:
        return "Few"
    if num_images > 31:
        return "Many"
    return "Medium"


def build_samples(
    core_annotation: dict[str, Any],
    data_dir: str | Path,
    dataset_name: str,
    adapter: BaseModelAdapter,
    max_context_len: int,
    tokens_per_image: int,
    combine_image: int | None = None,
    limit: int | None = None,
) -> list[MileBenchSample]:
    dataset_name = normalize_dataset_name(dataset_name)
    dataset_dir = Path(data_dir) / dataset_name
    img_dir = dataset_dir / "images"
    samples = []
    for ann in core_annotation["data"][:limit]:
        samples.append(
            build_sample(
                ann,
                core_annotation["meta_data"]["task_instruction"],
                img_dir,
                dataset_name,
                adapter,
                max_context_len,
                tokens_per_image,
                combine_image=combine_image,
            )
        )
    return samples


def build_sample(
    ann: dict[str, Any],
    task_instructions: dict[str, str],
    img_dir: Path,
    dataset_name: str,
    adapter: BaseModelAdapter,
    max_context_len: int,
    tokens_per_image: int,
    combine_image: int | None = None,
) -> MileBenchSample:
    task_instance = ann["task_instance"]
    task_instruction = task_instructions[ann["task_instruction_id"]]
    context = task_instance["context"]

    if "choice_list" in task_instance:
        choice_str = "\nChoice list: \n"
        choice_str += "\n".join(
            (f"{chr(65 + idx)}. " if "GPR1200" != dataset_name else "") + str(item)
            for idx, item in enumerate(task_instance["choice_list"])
        )
        choice_str += "\nYour answer is: "
        context += choice_str

    image_paths = _image_paths(task_instance, img_dir, combine_image)
    context = _replace_placeholders(context, len(image_paths), combine_image=combine_image)
    prompt, kept_image_paths = _truncate_context(
        task_instruction,
        context,
        image_paths,
        adapter,
        max_context_len,
        tokens_per_image,
        combine_image=combine_image,
    )
    return MileBenchSample(
        sample_id=int(ann["sample_id"]),
        prompt=prompt,
        image_paths=kept_image_paths,
        gt_response=str(ann["response"]),
        raw=ann,
    )


def _image_paths(task_instance: dict[str, Any], img_dir: Path, combine_image: int | None) -> list[str]:
    if combine_image:
        key = f"combined_{combine_image}_images"
        combined_dir = img_dir.parent / key
        return [str(combined_dir / path) for path in task_instance[key]]
    return [str(img_dir / path) for path in task_instance["images_path"]]


def _replace_placeholders(context: str, num_images: int, combine_image: int | None) -> str:
    for index in range(num_images):
        image_tag = "{image#%d}" % (index + 1)
        table_tag = "{table#%d}" % (index + 1)
        replacement = f"<Image {index + 1}> " if combine_image else IMAGE_PLACEHOLDER
        context = context.replace(image_tag, replacement)
        context = context.replace(table_tag, replacement)
    return context


def _truncate_context(
    task_instruction: str,
    context: str,
    image_paths: list[str],
    adapter: BaseModelAdapter,
    max_context_len: int,
    tokens_per_image: int,
    combine_image: int | None,
) -> tuple[str, list[str]]:
    if combine_image:
        prompt = f"{IMAGE_PLACEHOLDER}\n{task_instruction}\n{context}"
        return prompt, image_paths

    length_for_context = max(max_context_len - len(adapter.encode_text(task_instruction)), 1)
    raw_img_list = list(image_paths)
    fragments = context.split(IMAGE_PLACEHOLDER)[::-1]

    past_total_len = 0
    context_id_chunks: list[list[int]] = []
    kept_image_paths: list[str] = []
    image_start = False

    for fragment in fragments:
        cur_ids = adapter.encode_text(fragment)
        cur_len = len(cur_ids)
        if cur_len + past_total_len > length_for_context:
            if not context_id_chunks:
                context_id_chunks.insert(0, cur_ids[-length_for_context:])
            break

        image_start = False
        context_id_chunks.insert(0, cur_ids)
        past_total_len += cur_len

        if tokens_per_image + past_total_len > length_for_context:
            break
        if raw_img_list:
            image_start = True
            kept_image_paths.insert(0, raw_img_list.pop(-1))
            past_total_len += tokens_per_image

    if not context_id_chunks:
        ret_context = ""
    else:
        pieces = []
        for chunk in context_id_chunks[:-1]:
            pieces.append(adapter.decode_tokens(chunk))
            pieces.append(IMAGE_PLACEHOLDER)
        pieces.append(adapter.decode_tokens(context_id_chunks[-1]))
        ret_context = "".join(pieces)

    if image_start:
        ret_context = IMAGE_PLACEHOLDER + ret_context
    return f"{task_instruction}\n{ret_context}", kept_image_paths


def core_subset_for_predictions(core_annotation: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    ids = {int(item["sample_id"]) for item in predictions}
    subset = dict(core_annotation)
    subset["data"] = [sample for sample in core_annotation["data"] if int(sample["sample_id"]) in ids]
    return subset
