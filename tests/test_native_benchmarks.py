# SPDX-License-Identifier: Apache-2.0

import json

from PIL import Image

from mllm_kvcompress.benchmarks import BENCHMARKS, get_benchmark, parse_benchmark_list, run_benchmark, run_benchmarks
from mllm_kvcompress.benchmarks.aokvqa.data import aokvqa_sample
from mllm_kvcompress.benchmarks.data import load_sample_media
from mllm_kvcompress.benchmarks.eval import evaluate_predictions, parse_choice
from mllm_kvcompress.benchmarks.flickr30k.data import row_to_samples as flickr30k_row_to_samples
from mllm_kvcompress.benchmarks.milebench.eval import write_csv_summary
from mllm_kvcompress.benchmarks.mmmu.data import mmmu_sample
from mllm_kvcompress.benchmarks.native_runner import _check_data, parse_strategy_settings
from mllm_kvcompress.benchmarks.nocaps.data import caption_sample as nocaps_sample
from mllm_kvcompress.benchmarks.pca_bench.data import pca_sample


def tiny_image():
    return Image.new("RGB", (4, 4), "white")


def test_parse_benchmark_list_aliases():
    assert parse_benchmark_list("nocaps,filickr30k,A-OKVQA,MMMU,PCA-Bench,OCR-VQA") == [
        "nocaps",
        "flickr30k",
        "aokvqa",
        "mmmu",
        "pca_bench",
        "ocr_vqa",
    ]


def test_benchmark_registry_exposes_dataset_packages():
    assert sorted(BENCHMARKS) == ["aokvqa", "flickr30k", "mmmu", "nocaps", "ocr_vqa", "pca_bench"]
    assert get_benchmark("filickr30k").name == "flickr30k"
    assert get_benchmark("OCR-VQA").default_split == "test"


def test_strategy_all_uses_nonzero_ratios():
    names = [name for name, _ in parse_strategy_settings("all")]
    assert "baseline" in names
    assert "fastv_ratio0.5" in names
    assert "look_m_ratio0.5" in names
    assert "vidkv" in names


def test_aokvqa_native_sample_and_choice_eval():
    sample = aokvqa_sample(
        {
            "image": tiny_image(),
            "question_id": "q1",
            "question": "What color is it?",
            "choices": ["red", "white", "blue", "green"],
            "correct_choice_idx": 1,
        },
        0,
        "validation",
        None,
    )
    assert sample.answer == "B"
    assert sample.task_type == "multiple_choice"
    assert parse_choice("B", sample.choices or []) == 1
    metrics, rows = evaluate_predictions(
        "aokvqa",
        [
            {
                "sample_id": sample.sample_id,
                "task_type": sample.task_type,
                "choices": sample.choices,
                "answer_index": sample.answer_index,
                "pred_response": "B",
            }
        ],
    )
    assert metrics["Accuracy"] == 1.0
    assert rows[0]["extracted"] == "B"


def test_caption_sample_uses_native_references():
    sample = flickr30k_row_to_samples(
        {"image": tiny_image(), "img_id": "1", "caption": ["a white square", "blank image"]},
        0,
        "test",
        None,
    )[0]
    metrics, _ = evaluate_predictions(
        "flickr30k",
        [{"sample_id": sample.sample_id, "task_type": "caption", "references": sample.references, "pred_response": "a white square"}],
    )
    assert metrics["Rouge-L f"] > 0.99


def test_local_native_json_resolves_relative_image_paths(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_path = images_dir / "sample.jpg"
    tiny_image().save(image_path)
    data_path = tmp_path / "validation.json"
    data_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "n0",
                    "image": "sample.jpg",
                    "annotations_captions": ["a white square"],
                }
            ]
        ),
        encoding="utf-8",
    )
    sample = get_benchmark("nocaps").load_samples(source=data_path, limit=1)[0]
    assert sample.media == [str(image_path)]
    assert load_sample_media(sample)[0].size == (4, 4)


def test_mmmu_replaces_numbered_image_tags():
    sample = mmmu_sample(
        {
            "id": "m1",
            "question": "<image 1> Choose the answer.",
            "options": "['alpha', 'beta']",
            "answer": "A",
            "question_type": "multiple-choice",
            "image_1": tiny_image(),
        },
        0,
        "validation",
        "Accounting",
    )
    assert "<image 1>" not in sample.prompt
    assert sample.prompt.count("<ImageHere>") == 1
    assert sample.answer_index == 0


def test_pca_hidden_label_is_unlabeled():
    sample = pca_sample(
        {
            "image": tiny_image(),
            "question": "What is the best action?",
            "actions": ["go", "stop"],
            "answer_index": -1,
        },
        0,
        "test_closed",
        "Autonomous Driving",
    )
    metrics, rows = evaluate_predictions(
        "pca_bench",
        [
            {
                "sample_id": sample.sample_id,
                "task_type": sample.task_type,
                "choices": sample.choices,
                "answer_index": sample.answer_index,
                "pred_response": "A",
            }
        ],
    )
    assert metrics["num_labeled"] == 0
    assert metrics["num_unlabeled"] == 1
    assert rows[0]["score"] == ""


def test_csv_summary_skips_non_numeric_metrics(tmp_path):
    path = tmp_path / "result.csv"
    write_csv_summary(
        path,
        {
            "model": {
                "Native Benchmarks": {
                    "nocaps": {
                        "Rouge-L f": 0.5,
                        "CIDEr unavailable": "Install pycocoevalcap for official caption scoring.",
                    }
                }
            }
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "nocaps (Rouge-L f)" in text
    assert "CIDEr unavailable" not in text


def test_check_data_writes_first_media_report(tmp_path):
    image_path = tmp_path / "sample.jpg"
    tiny_image().save(image_path)
    sample = nocaps_sample(
        {"image": str(image_path), "image_id": "n0", "annotations_captions": ["a white square"]},
        0,
        "validation",
        None,
    )
    report = _check_data(tmp_path / "out", {"nocaps": [sample]})
    assert report["nocaps"]["first_media_load"]["ok"] is True
    assert (tmp_path / "out" / "data_check.json").exists()


def test_public_api_runs_data_check_without_loading_model(tmp_path):
    image_path = tmp_path / "sample.jpg"
    tiny_image().save(image_path)
    data_path = tmp_path / "nocaps.json"
    data_path.write_text(
        json.dumps([{"image": str(image_path), "image_id": "n0", "annotations_captions": ["a white square"]}]),
        encoding="utf-8",
    )
    report = run_benchmarks(
        model="unused",
        benchmarks="nocaps",
        settings="baseline",
        sources={"nocaps": str(data_path)},
        output_dir=tmp_path / "run",
        check_data_only=True,
    )
    assert report["nocaps"]["first_media_load"]["ok"] is True


def test_public_api_accepts_method_kwargs(tmp_path):
    image_path = tmp_path / "sample.jpg"
    tiny_image().save(image_path)
    data_path = tmp_path / "nocaps.json"
    data_path.write_text(
        json.dumps([{"image": str(image_path), "image_id": "n0", "annotations_captions": ["a white square"]}]),
        encoding="utf-8",
    )
    report = run_benchmark(
        model="unused",
        benchmark="nocaps",
        method="look_m",
        method_kwargs={"ratio": 0.5},
        sources={"nocaps": str(data_path)},
        output_dir=tmp_path / "run",
        check_data_only=True,
    )
    assert report["nocaps"]["first_media_load"]["ok"] is True
