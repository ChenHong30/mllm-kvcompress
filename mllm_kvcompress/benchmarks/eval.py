# SPDX-License-Identifier: Apache-2.0
"""Evaluation helpers for native benchmarks."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from mllm_kvcompress.benchmarks.milebench.eval import _rouge_l_fallback, mean


def evaluate_predictions(benchmark: str, predictions: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not predictions:
        return {"num_samples": 0}, []

    task_type = predictions[0].get("task_type", "")
    if benchmark in {"nocaps", "flickr30k"} or task_type == "caption":
        return _evaluate_caption(predictions)
    if benchmark == "ocr_vqa" or task_type == "ocr_vqa":
        return _evaluate_ocr_vqa(predictions)
    if any(item.get("task_type") == "multiple_choice" for item in predictions):
        return _evaluate_mixed(predictions)
    return _evaluate_open(predictions)


def _evaluate_caption(predictions: list[dict[str, Any]]):
    refs = {str(i): item.get("references", []) for i, item in enumerate(predictions)}
    hyps = {str(i): [item.get("pred_response", "")] for i, item in enumerate(predictions)}
    metrics = {"num_samples": len(predictions)}
    eval_list = []

    cider_score, cider_scores = _cider(refs, hyps)
    if cider_score is not None:
        metrics["CIDEr"] = float(cider_score)
    rouge_scores = []
    for idx, item in enumerate(predictions):
        rouge = _best_rouge(item.get("pred_response", ""), item.get("references", []))
        rouge_scores.append(rouge)
        row = {"id": str(item.get("sample_id", idx)), "Rouge-L f": round(rouge, 4)}
        if cider_scores is not None:
            row["CIDEr"] = round(float(cider_scores[idx]), 4)
        eval_list.append(row)
    metrics["Rouge-L f"] = mean(rouge_scores)
    if cider_score is None:
        metrics["CIDEr unavailable"] = "Install pycocoevalcap for official caption scoring."
    return metrics, eval_list


def _evaluate_ocr_vqa(predictions: list[dict[str, Any]]):
    scores = []
    eval_list = []
    for item in predictions:
        score = _best_rouge(item.get("pred_response", ""), item.get("references", []))
        scores.append(score)
        eval_list.append({"id": str(item.get("sample_id", "")), "score": round(score, 4)})
    return {"Rouge-L f": mean(scores), "num_samples": len(predictions)}, eval_list


def _evaluate_mixed(predictions: list[dict[str, Any]]):
    by_type: dict[str, list[float]] = defaultdict(list)
    eval_list = []
    unlabeled = 0
    for item in predictions:
        if item.get("task_type") == "multiple_choice":
            pred_index = parse_choice(item.get("pred_response", ""), item.get("choices") or [])
            answer_index = item.get("answer_index")
            if answer_index is None:
                answer_index = _letter_to_index(str(item.get("answer", "")))
            if answer_index is None:
                unlabeled += 1
                eval_list.append(
                    {
                        "id": str(item.get("sample_id", "")),
                        "score": "",
                        "extracted": _letter(pred_index),
                        "answer": "",
                    }
                )
                continue
            score = float(pred_index == answer_index)
            metric_name = "Accuracy"
            item_result = {"extracted": _letter(pred_index), "answer": _letter(answer_index)}
        else:
            score = _best_rouge(item.get("pred_response", ""), item.get("references", []))
            metric_name = "Rouge-L f"
            item_result = {}
        by_type[metric_name].append(score)
        eval_row = {"id": str(item.get("sample_id", "")), "score": round(score, 4), **item_result}
        eval_list.append(eval_row)
    metrics = {metric: mean(values) for metric, values in by_type.items()}
    metrics["num_samples"] = len(predictions)
    metrics["num_labeled"] = len(predictions) - unlabeled
    if unlabeled:
        metrics["num_unlabeled"] = unlabeled
    return metrics, eval_list


def _evaluate_open(predictions: list[dict[str, Any]]):
    scores = []
    eval_list = []
    for item in predictions:
        score = _best_rouge(item.get("pred_response", ""), item.get("references", []))
        scores.append(score)
        eval_list.append({"id": str(item.get("sample_id", "")), "score": round(score, 4)})
    return {"Rouge-L f": mean(scores), "num_samples": len(predictions)}, eval_list


def parse_choice(answer: str, choices: list[str]) -> int | None:
    text = str(answer).strip()
    if not text:
        return None
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return ord(match.group(1)) - ord("A")
    match = re.search(r"\(([A-Z])\)", text.upper())
    if match:
        return ord(match.group(1)) - ord("A")
    lowered = _normalize_text(text)
    for index, choice in enumerate(choices):
        if _normalize_text(choice) in lowered:
            return index
    return None


def _best_rouge(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    pred = _normalize_text(prediction)
    return max(_rouge_l_fallback(pred, _normalize_text(ref)) for ref in references)


def _cider(refs: dict[str, list[str]], hyps: dict[str, list[str]]):
    try:
        from pycocoevalcap.cider.cider import Cider

        score, scores = Cider().compute_score(refs, hyps)
        return score, scores
    except Exception:
        return None, None


def _normalize_text(text: Any) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _letter(index: int | None) -> str:
    if index is None or index < 0:
        return ""
    return chr(ord("A") + index) if index < 26 else str(index)


def _letter_to_index(value: str) -> int | None:
    value = value.strip().upper()
    if len(value) == 1 and "A" <= value <= "Z":
        return ord(value) - ord("A")
    return None
