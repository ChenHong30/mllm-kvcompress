# SPDX-License-Identifier: Apache-2.0
"""Official-style MileBench scoring."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from mllm_kvcompress.benchmarks.milebench.data import image_quantity_level


class MileBenchEvaluator:
    """Evaluator matching the public MileBench implementation."""

    def __init__(self):
        self.period_strip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.comma_strip = re.compile(r"(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def char(self, index: int) -> str:
        if index < 26:
            return chr(index + 65)
        if index < 52:
            return "A" + chr(index + 65 - 26)
        return "B" + chr(index + 65 - 26 - 26)

    def process_punctuation(self, text: str) -> str:
        output = text
        for punct in self.punct:
            if (punct + " " in text or " " + punct in text) or re.search(self.comma_strip, text) is not None:
                output = output.replace(punct, "")
            else:
                output = output.replace(punct, " ")
        return self.period_strip.sub("", output, re.UNICODE)

    def process(self, answer: Any) -> str:
        answer = str(answer).replace("\n", " ").replace("\t", " ").strip()
        answer = self.process_punctuation(answer)
        answer = answer.strip("'").strip('"').strip().lower()
        return answer

    def evaluate(self, predictions: list[dict[str, Any]], core_json: dict[str, Any], dataset_name: str):
        predictions = _attach_core_fields(predictions, core_json)
        question_type = core_json["meta_data"]["question_type"]
        if "NeedleInAHaystack" in dataset_name or "MMCoQA" in dataset_name:
            return self.evaluate_needle(predictions, needle="NeedleInAHaystack" in dataset_name)
        if question_type == "open-ended":
            return self.evaluate_rouge(predictions)
        if question_type == "multi-choice":
            predictions_with_extracted, result, eval_list = self.evaluate_multichoice(predictions)
            return result, eval_list, predictions_with_extracted
        raise ValueError(f"Unsupported MileBench question type: {question_type}")

    def evaluate_rouge(self, predictions: list[dict[str, Any]]):
        scores = []
        eval_list = []
        by_level = {"Few": [], "Medium": [], "Many": []}
        for item in predictions:
            gt_ans = self.process(item["gt_response"])
            pred_ans = self.process(item["pred_response"])
            score = 0.0 if pred_ans == "" else rouge_l_f(pred_ans, gt_ans)
            scores.append(score)
            by_level[item["image_quantity_level"]].append(score)
            eval_list.append({"id": str(item["sample_id"]), "score": str(round(score, 3))})
        return {
            "Rouge-L f": mean(scores),
            "image_quantity_level-Accuracy": {key: mean(value) for key, value in by_level.items()},
            "image_quantity_level-Result": {key: [sum(value), len(value)] for key, value in by_level.items()},
        }, eval_list, None

    def match_choice(self, text: str, option: dict[str, str]) -> str:
        def preprocess_option_string(option_string: str) -> str:
            processed_option = self.process(option_string)
            for char in ["\\", ".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "|", "(", ")"]:
                processed_option = processed_option.replace(char, "\\" + char)
            return processed_option

        if text == "":
            return "C"
        try:
            option_str = "|".join(preprocess_option_string(f"{key} {value}") for key, value in option.items())
            option_res = re.search(rf"({option_str})", text, re.S)
            if option_res:
                return option_res.group(0)[0].upper()

            option_str = "|".join(preprocess_option_string(value).replace(" ", "") for value in option.values())
            option_res = re.search(rf"({option_str})", text.replace(" ", ""), re.S)
            if option_res:
                for key, value in option.items():
                    if option_res[0].strip() == preprocess_option_string(value).replace(" ", ""):
                        return key.upper()

            if len(text) in [1, 2] and text.upper() in option:
                return text.upper()
        except Exception:
            return text
        return "".join(char.upper() for char in text if char.upper() in option)

    def evaluate_multichoice(self, predictions: list[dict[str, Any]]):
        correct = 0
        eval_list = []
        by_level = {"Few": [], "Medium": [], "Many": []}
        for sample in predictions:
            sample["gt_response"] = self.process(sample["gt_response"])
            sample["pred_response"] = self.process(sample["pred_response"])
            sample["choice_list"] = [self.process(choice) for choice in sample["choice_list"]]

            option_dict = {self.char(index): choice for index, choice in enumerate(sample["choice_list"])}
            selected_answer = self.match_choice(sample["pred_response"], option_dict)
            gt_ans_chr = self.char(sample["choice_list"].index(sample["gt_response"]))
            score = int(selected_answer == gt_ans_chr)

            sample["extracted"] = selected_answer
            sample["result"] = score
            eval_list.append({"id": str(sample["sample_id"]), "score": str(score)})
            correct += score
            by_level[sample["image_quantity_level"]].append(score)
        return predictions, {
            "Accuracy": correct / len(predictions) if predictions else 0.0,
            "image_quantity_level-Accuracy": {key: mean(value) for key, value in by_level.items()},
            "image_quantity_level-Result": {key: [sum(value), len(value)] for key, value in by_level.items()},
        }, eval_list

    def evaluate_needle(self, predictions: list[dict[str, Any]], needle: bool = True):
        correct = 0
        eval_list = []
        by_level = {"Few": [], "Medium": [], "Many": []}
        for sample in predictions:
            gt_ans = self.process(sample["gt_response"])
            pred_ans = self.process(sample["pred_response"])
            score = int(gt_ans in pred_ans.split()) if needle else int(gt_ans in pred_ans)
            sample["result"] = score
            eval_list.append({"id": str(sample["sample_id"]), "score": str(score)})
            correct += score
            by_level[sample["image_quantity_level"]].append(score)
        return {
            "Accuracy": correct / len(predictions) if predictions else 0.0,
            "image_quantity_level-Accuracy": {key: mean(value) for key, value in by_level.items()},
            "image_quantity_level-Result": {key: [sum(value), len(value)] for key, value in by_level.items()},
        }, eval_list, None


def _attach_core_fields(predictions: list[dict[str, Any]], core_json: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {int(item["sample_id"]): dict(item) for item in predictions}
    missing = []
    for sample in core_json["data"]:
        sample_id = int(sample["sample_id"])
        if sample_id not in by_id:
            missing.append(sample_id)
            continue
        pred = by_id[sample_id]
        task_instance = sample["task_instance"]
        pred["choice_list"] = task_instance.get("choice_list", pred.get("choice_list"))
        pred["image"] = task_instance["images_path"]
        pred["image_quantity_level"] = sample.get(
            "image_quantity_level", image_quantity_level(len(task_instance["images_path"]))
        )
        by_id[sample_id] = pred
    if missing and len(missing) == len(core_json["data"]):
        raise ValueError("No predictions match the MileBench annotation sample ids")
    return [by_id[int(item["sample_id"])] for item in predictions]


def rouge_l_f(prediction: str, target: str) -> float:
    try:
        from rouge import Rouge

        return Rouge().get_scores(prediction, target)[0]["rouge-l"]["f"]
    except Exception:
        return _rouge_l_fallback(prediction, target)


def _rouge_l_fallback(prediction: str, target: str) -> float:
    pred_tokens = prediction.split()
    target_tokens = target.split()
    if not pred_tokens or not target_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, target_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(target_tokens)
    if precision == 0.0 or recall == 0.0:
        return 0.0
    beta = precision / (recall + 1e-12)
    return ((1 + beta**2) * recall * precision) / (recall + beta**2 * precision + 1e-12)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def write_json(path: str | Path, payload: Any, indent: int | None = 4):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_results(output_dir: str | Path, model_names: list[str]) -> dict[str, Any]:
    from mllm_kvcompress.benchmarks.milebench.data import DATASET_GROUPS

    output_dir = Path(output_dir)
    result = {}
    for model_name in model_names:
        model_result = {}
        for group, datasets in DATASET_GROUPS.items():
            task_result = {}
            for dataset in datasets:
                eval_path = output_dir / model_name / dataset / "eval.json"
                task_result[dataset] = read_json(eval_path) if eval_path.exists() else {}
            model_result[group] = task_result
        result[model_name] = model_result
    write_json(output_dir / "result.json", result)
    write_csv_summary(output_dir / "result.csv", result)
    return result


def write_csv_summary(path: str | Path, result: dict[str, Any]):
    rows = []
    columns = ["Model"]
    ignored = {"image_quantity_level-Accuracy", "image_quantity_level-Result", "Diff-Accuracy"}
    for model_name, groups in result.items():
        row = {"Model": model_name}
        for datasets in groups.values():
            for dataset, metrics in datasets.items():
                for metric, value in metrics.items():
                    if metric in ignored or isinstance(value, dict):
                        continue
                    column = f"{dataset} ({metric})"
                    if column not in columns:
                        columns.append(column)
                    row[column] = round(float(value) * 100, 2) if not math.isnan(float(value)) else ""
        rows.append(row)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(columns) + "\n")
        for row in rows:
            handle.write(",".join(_csv_cell(row.get(column, "")) for column in columns) + "\n")


def _csv_cell(value: Any) -> str:
    text = str(value)
    if any(char in text for char in [",", '"', "\n"]):
        text = '"' + text.replace('"', '""') + '"'
    return text
