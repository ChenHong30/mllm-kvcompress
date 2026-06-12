# SPDX-License-Identifier: Apache-2.0
"""
Evaluate compression methods on the A-OKVQA validation set (multiple choice), the
benchmark used by the FastV paper (Table 1). Reference numbers on LLaVA-1.5-7B reported
by FastV: baseline 76.7, K=2 R=50% -> 77.0 (lossless), K=2 R=75% -> 75.5 (-1.2).

Usage:
    python examples/eval_aokvqa.py <validation.parquet> <model_path> [n_samples]
"""

import io
import json
import sys
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mllm_kvcompress import (
    CompressedContextPipeline,
    FastV,
    FitPrune,
    GUIKV,
    InfiniPotV,
    LookM,
    MEDA,
    MixKV,
    SparseMM,
    STaRKV,
    VidKV,
)

LETTERS = ["A", "B", "C", "D"]
PROMPT = "{question}\n{options}\nAnswer with the option's letter from the given choices directly."


def parse_answer(answer: str, choices) -> int:
    answer = answer.strip()
    if answer and answer[0] in LETTERS:
        return LETTERS.index(answer[0])
    for i, letter in enumerate(LETTERS[: len(choices)]):
        if f"{letter}." in answer or f"({letter})" in answer:
            return i
    lowered = answer.lower()
    for i, choice in enumerate(choices):
        if choice.lower() in lowered:
            return i
    return -1


def main():
    parquet_path, model_path = sys.argv[1], sys.argv[2]
    n_samples = int(sys.argv[3]) if len(sys.argv) > 3 else None

    df = pd.read_parquet(parquet_path)
    if n_samples:
        df = df.iloc[:n_samples]

    model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16).to("cuda")
    processor = AutoProcessor.from_pretrained(model_path)
    pipe = CompressedContextPipeline(model=model, processor=processor)

    settings = {
        "baseline": None,
        "fastv_0.5": FastV(ratio=0.5, filter_layer=2),
        "fastv_0.75": FastV(ratio=0.75, filter_layer=2),
        "lookm_0.5": LookM(ratio=0.5),
        "lookm_0.5_nomerge": LookM(ratio=0.5, merge="none"),
        "meda_0.5": MEDA(ratio=0.5),
        "mixkv_0.5": MixKV(ratio=0.5),
        "sparsemm_0.5": SparseMM(ratio=0.5),
        "guikv_0.5": GUIKV(ratio=0.5),
        "starkv_0.5": STaRKV(ratio=0.5),
        "infinipotv_0.5": InfiniPotV(ratio=0.5),
        "vidkv_1.5bit": VidKV(key_bits=1.5, value_bits=1.58),
        "fitprune_0.5": FitPrune(ratio=0.5),
    }

    results = {}
    for name, method in settings.items():
        correct, start = 0, time.time()
        for i, row in enumerate(df.itertuples()):
            image = Image.open(io.BytesIO(row.image["bytes"])).convert("RGB")
            options = "\n".join(f"{LETTERS[j]}. {c}" for j, c in enumerate(row.choices))
            question = PROMPT.format(question=row.question, options=options)
            out = pipe(image, question=question, method=method, max_new_tokens=5)
            pred = parse_answer(out["answer"], row.choices)
            correct += int(pred == row.correct_choice_idx)
            if (i + 1) % 100 == 0:
                print(
                    f"[{name}] {i + 1}/{len(df)} acc={correct / (i + 1):.4f} "
                    f"({(time.time() - start) / (i + 1):.2f}s/sample)",
                    flush=True,
                )
        results[name] = correct / len(df)
        print(f"== {name}: accuracy {results[name]:.4f} ({correct}/{len(df)}) ==", flush=True)
        with open("/tmp/aokvqa_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
