# SPDX-License-Identifier: Apache-2.0
"""
Quick demo of mllm_kvcompress on Qwen2-VL-2B-Instruct (runs on a single 8GB GPU).

    python examples/demo.py [model_name_or_path]
"""

import sys

import requests
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from mllm_kvcompress import CompressedContextPipeline, FastV, LookM, MEDA

model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-2B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(model_name)
pipe = CompressedContextPipeline(model=model, processor=processor)

url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
question = "Describe this image in one sentence."

for method in [
    None,
    LookM(ratio=0.5),
    MEDA(ratio=0.5),
    FastV(ratio=0.5),
]:
    name = method.__class__.__name__ if method else "no compression"
    result = pipe(image, question=question, method=method, max_new_tokens=50)
    print(f"\n=== {name} ===\n{result['answer']}")
