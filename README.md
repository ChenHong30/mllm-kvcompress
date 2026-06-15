<div align="center">

![mllm-kvcompress](banner.png)

**KV cache compression for multimodal LLMs, made easy.**

[English](README.md) | [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![transformers](https://img.shields.io/badge/transformers-4.55%2B%20%7C%205.x-orange.svg)](https://github.com/huggingface/transformers)

</div>

---

Multimodal LLMs inject thousands of vision tokens into the pre-fill stage, making the KV cache a major
inference bottleneck. Research on MLLM-specific KV cache compression is growing fast, but every method ships
as a standalone fork of a model codebase. **mllm-kvcompress** aggregates these methods behind a single,
hook-based interface: one compression method = one class = one Python file, rewritten for multimodal
architectures (modality-aware scoring, multimodal rotary embeddings, per-layer budgets).

## Installation

```bash
pip install -e .
```

This installs all runtime dependencies (`torch`, `transformers`, `pillow`). Optional extras:

```bash
pip install -e ".[dev]"       # + pytest, for running the test suite
pip install -e ".[internvl]"  # + einops, timm, torchvision and transformers 4.x, for InternVL3
pip install -e ".[bench]"     # + benchmark helpers (datasets, rouge, tqdm, ...)
```

The `internvl` extra pins `transformers<5` because the InternVL3 remote code is incompatible with
transformers 5.x; all other supported models work with any transformers in the declared range.

## Supported environments

| Dependency | Versions |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.3 |
| transformers | one dependency range: ≥ 4.55, < 6 (4.55 and 5.11 tested) |
| Attention backends | sdpa (default), eager |

A 2B model (e.g. Qwen2-VL-2B-Instruct, bf16) runs end-to-end within 8 GB of VRAM.
Qwen3-VL and Qwen3.5 require a transformers build that provides `qwen3_vl` / `qwen3_5` (tested with 5.11.0).

## Supported models

| Model family | Classes |
|---|---|
| Qwen2-VL / Qwen2.5-VL | `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration` |
| Qwen3-VL | `Qwen3VLForConditionalGeneration` |
| Qwen3.5 native multimodal | `Qwen3_5ForConditionalGeneration` |
| InternVL3 | `InternVLChatModel` |

InternVL3 checkpoints use custom remote code and do not ship a HuggingFace multimodal `Processor`; use their
`chat`/`generate` preprocessing path and wrap it with `compress()`. Their remote code is currently incompatible
with transformers 5.x (it fails at load time), so use a 4.x environment. The bundled evaluation pipeline remains
processor-based and is intended for Qwen-style `AutoModelForImageTextToText` models.

## Supported compression methods

All methods are training-free and applied during pre-filling only.

| Method       | Source                                                                                                                                                                                                                   | Idea                                                                     |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `LookM`      | [Paper](https://aclanthology.org/2024.findings-emnlp.235/) · [Code](https://github.com/SUSTechBruce/LOOK-M) (Findings of EMNLP 2024)                                                                                     | Text-prior eviction + merging of evicted visual KVs                      |
| `MEDA`       | [Paper](https://aclanthology.org/2025.naacl-long.125/) · [Code](https://github.com/aiot-mlsys-lab/meda) (NAACL 2025)                                                                                                     | Layer-wise dynamic budgets via multimodal attention entropy              |
| `FastV`      | [Paper](https://arxiv.org/abs/2403.06764) · [Code](https://github.com/pkunlp-icler/FastV) (ECCV 2024 Oral)                                                                                                               | Vision token pruning after an early filter layer                         |
| `FitPrune`   | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34366) · [Code](https://github.com/ywh187/FitPrune) (AAAI 2025)                                                                                                 | Progressive vision token pruning with a per-layer schedule               |
| `SparseMM`   | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SparseMM_Head_Sparsity_Emerges_from_Visual_Concept_Responses_in_MLLMs_ICCV_2025_paper.html) · [Code](https://github.com/CR400AF-A/SparseMM) (ICCV 2025) | Head-wise budgets driven by visual-head importance                       |
| `MixKV`      | [Paper](https://arxiv.org/abs/2510.20707) · [Code](https://github.com/xuyang-liu16/MixKV) (ICLR 2026 Poster)                                                                                                             | Mixing importance and diversity scores per head                          |
| `GUIKV`      | [Paper](https://arxiv.org/abs/2510.00536) · [Code](https://github.com/SalesforceAIResearch/GUI-KV)                                                                                                                       | Spatial saliency + temporal redundancy, for GUI agents                   |
| `STaRKV`     | [Paper](https://arxiv.org/abs/2606.01790) · [Code](https://github.com/kawhiiiileo/STaR-KV)                                                                                                                               | Spatial MI prior + entropy-guided score sharpening                       |
| `InfiniPotV` | [Paper](https://openreview.net/forum?id=hFxOZjHyTg) · [Code](https://github.com/aiha-lab/InfiniPot-V) (NeurIPS 2025 Poster)                                                                                              | Vision-only eviction (temporal redundancy + value norm), streaming video |
| `VidKV`      | [Paper](https://arxiv.org/abs/2503.16257) · [Code](https://github.com/KD-TAO/VidKV)                                                                                                                                      | ~1.x-bit mixed-precision KV quantization of vision tokens                |


Deviations from the official implementations (and the measurements behind them) are documented in each
method's docstring.

## Quick start

Wrap any forward pass or `generate` call with `compress(model, method)`:

```python
import requests
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from mllm_kvcompress import compress, LookM

model_name = "Qwen/Qwen2-VL-2B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(model_name, dtype="bfloat16").to("cuda")
processor = AutoProcessor.from_pretrained(model_name)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe the image."}]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

with compress(model, LookM(ratio=0.5), verbose=True):
    outputs = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

With `verbose=True`, the measured KV cache sizes are printed after inference — measured on the actual
cache, not derived from the requested `ratio` (the two never match exactly, e.g. because text tokens or
recent windows are protected):

```
[mllm_kvcompress] LookM compression summary (28 layers)
  KV cache before: 281 tokens/layer, 7.68 MiB
  KV cache after:  140 tokens/layer, 3.83 MiB
  measured compression ratio: 50.2%
```

Methods with per-head budgets keep the cache size unchanged and neutralize evicted keys during decoding
instead; for those the summary additionally reports the measured *virtual* compression ratio.

Local checkpoints under `model_weight/` can be passed directly, for example
`model_weight/Qwen3-VL-2B-Thinking` or `model_weight/Qwen3.5-0.8B`, provided your transformers version includes
those architectures.

Methods can also be created by name from the registry:

```python
from mllm_kvcompress import create_method, METHODS

print(sorted(METHODS))  # ['fastv', 'fitprune', 'gui_kv', ...]
method = create_method("sparsemm", ratio=0.5)
```

## Advanced usage

### Evaluation pipeline: many questions, one compressed context

For evaluation, the question should not be compressed away with the context — a compressed cache must work
for *any* follow-up query. The bundled pipeline pre-fills (and compresses) the multimodal context once, then
answers each question on the compressed cache:

```python
import requests
from PIL import Image
from transformers import pipeline
import mllm_kvcompress  # registers the pipeline
from mllm_kvcompress import MEDA

pipe = pipeline("multimodal-kv-compression", model="Qwen/Qwen2-VL-2B-Instruct", dtype="bfloat16", device="cuda")

base = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main"
image1 = Image.open(requests.get(f"{base}/bee.jpg", stream=True).raw)
image2 = Image.open(requests.get(f"{base}/pipeline-cat-chonk.jpeg", stream=True).raw)

result = pipe(
    [image1, image2],
    context="The first photo was taken in a garden, the second one in the snow.",
    questions=["What insect is in the first photo?", "What animal is in the second photo?"],
    method=MEDA(ratio=0.5),
    max_new_tokens=50,
    verbose=True,  # print the measured compression summary after pre-filling
)
print(result["answers"])
```

## Benchmark

```bash
pip install -e ".[bench]"
```

The benchmark interface is framework-level and independent of any compression method.
The current native benchmark set includes `nocaps`, `flickr30k`, `aokvqa`, `mmmu`,
`pca_bench`, and `ocr_vqa`; more benchmarks will be added gradually.

Choose a model, one or more compression settings, and one or more benchmarks:

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --settings baseline,look_m_0.5,fastv_0.5 \
  --benchmarks nocaps,aokvqa \
  --limit 100 \
  --output-dir runs/qwen2vl_native
```

Use `--settings all` to run the full-cache baseline plus every registered compression
strategy at a standard 50% setting where applicable (`VidKV` uses its native bit-width
defaults). The native runner uses Hugging Face sources by default:
`HuggingFaceM4/NoCaps`, `lmms-lab/flickr30k`, `HuggingFaceM4/A-OKVQA`, `MMMU/MMMU`,
`PCA-Bench/PCA-Bench-V1`, and `howard-hou/OCR-VQA`.

Override any dataset source with a local path or another Hugging Face id:

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model /path/to/model \
  --settings baseline,look_m_0.5 \
  --benchmarks aokvqa,ocr_vqa \
  --source aokvqa=/data/hkustgz/datasets/A-OKVQA \
  --source ocr_vqa=/path/to/native/ocr-vqa.json \
  --output-dir runs/local_sources
```

Use `--check-data-only --limit 1` to validate samples and first-media loading before
paying the model-loading cost:

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model /path/to/model \
  --benchmarks nocaps \
  --check-data-only \
  --limit 1
```

The same interface is available as a Python API:

```python
from mllm_kvcompress.benchmarks import run_benchmark

result = run_benchmark(
    model="Qwen/Qwen2-VL-2B-Instruct",
    benchmark="aokvqa",
    method="look_m",
    method_kwargs={"ratio": 0.5},
    limit=100,
    output_dir="runs/aokvqa_lookm",
)
```

The benchmark registry mirrors the method registry: each dataset lives under
`mllm_kvcompress/benchmarks/<benchmark>/`, and each compression strategy is selected
from `mllm_kvcompress.methods`.

PCA-Bench's public `test_closed` split may hide labels (`answer_index=-1`); predictions
are still produced, and evaluation reports unlabeled counts unless a labeled local
source is supplied.

## Acknowledgements

This project is inspired by [NVIDIA kvpress](https://github.com/NVIDIA/kvpress), which pioneered the
hook-based, one-file-per-method approach to KV cache compression for text-only LLMs.

We are grateful to the authors of the open-source MLLM KV cache compression methods aggregated here:
[LOOK-M](https://github.com/SUSTechBruce/LOOK-M), [MEDA](https://github.com/AIoT-MLSys-Lab/MEDA),
[FastV](https://github.com/pkunlp-icler/fastv), [FitPrune](https://github.com/ywh187/FitPrune),
[SparseMM](https://github.com/CR400AF-A/SparseMM), [MixKV](https://github.com/xuyang-liu16/MixKV),
[GUI-KV](https://github.com/SalesforceAIResearch/GUI-KV), [STaR-KV](https://github.com/kawhiiiileo/STaR-KV),
[InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V) and [VidKV](https://github.com/KD-TAO/VidKV).

## Citation

If this project is helpful to your work, please cite it as:

```bibtex
@software{mllm_kvcompress,
  author = {Chen, Hong},
  title  = {MLLM-KVCompress: KV cache compression for multimodal LLMs},
  url    = {https://github.com/ChenHong30/MLLM-KVCompress},
  year   = {2026}
}
```

Please also cite the original papers of the compression methods you use (see the
[method table](#supported-compression-methods) for references).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChenHong30/MLLM-KVCompress&type=Date)](https://star-history.com/#ChenHong30/MLLM-KVCompress&Date)
