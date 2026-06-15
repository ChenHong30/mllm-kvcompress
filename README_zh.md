<div align="center">

![mllm-kvcompress](banner.png)

**KV cache compression for multimodal LLMs, made easy.**

[English](README.md) | [简体中文](README_zh.md)

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![transformers](https://img.shields.io/badge/transformers-4.55%2B%20%7C%205.x-orange.svg)](https://github.com/huggingface/transformers)

</div>

---

本中文文档由 GPT-5.5 翻译而来，如有歧义，以英文版为准。

多模态大模型会在 pre-fill 阶段注入大量视觉 token，KV cache 很容易成为推理瓶颈。
面向 MLLM 的 KV cache 压缩研究正在快速发展，但很多方法都以模型代码 fork 的形式发布。
**mllm-kvcompress** 把这些方法统一到一个 hook-based 接口下：
一个压缩方法 = 一个类 = 一个 Python 文件，并针对多模态架构重新实现了模态感知打分、
多模态 RoPE、分层预算等逻辑。

## 安装

```bash
pip install -e .
```

这会安装运行时依赖：`torch`、`transformers`、`pillow`。可选依赖如下：

```bash
pip install -e ".[dev]"       # + pytest，用于运行测试
pip install -e ".[internvl]"  # + einops、timm、torchvision 和 transformers 4.x，用于 InternVL3
pip install -e ".[bench]"     # + benchmark 相关依赖，例如 datasets、rouge、tqdm
```

`internvl` extra 会限制 `transformers<5`，因为 InternVL3 的 remote code 目前不兼容
transformers 5.x。其他支持的模型可以使用项目声明范围内的 transformers 版本。

## 支持环境

| 依赖 | 版本 |
|---|---|
| Python | ≥ 3.10 |
| PyTorch | ≥ 2.3 |
| transformers | 统一依赖范围：≥ 4.55, < 6，已测试 4.55 和 5.11 |
| Attention backend | sdpa 默认，eager |

2B 级别模型，例如 Qwen2-VL-2B-Instruct bf16，可以在 8 GB 显存内端到端运行。
Qwen3-VL 和 Qwen3.5 需要包含 `qwen3_vl` / `qwen3_5` 架构的 transformers 版本，
当前已用 5.11.0 测试。

## 支持模型

| 模型家族 | 类 |
|---|---|
| Qwen2-VL / Qwen2.5-VL | `Qwen2VLForConditionalGeneration`, `Qwen2_5_VLForConditionalGeneration` |
| Qwen3-VL | `Qwen3VLForConditionalGeneration` |
| Qwen3.5 native multimodal | `Qwen3_5ForConditionalGeneration` |
| InternVL3 | `InternVLChatModel` |

InternVL3 checkpoint 使用自定义 remote code，并且不提供 HuggingFace 多模态 `Processor`；
需要使用它自己的 `chat`/`generate` 预处理路径，然后用 `compress()` 包起来。
它的 remote code 目前不兼容 transformers 5.x，因此建议使用 4.x 环境。
内置评测 pipeline 仍然是 processor-based，主要面向 Qwen 风格的
`AutoModelForImageTextToText` 模型。

## 支持压缩方法

所有方法都不需要训练，并且只在 pre-fill 阶段应用。

| 方法 | 来源 | 核心思路 |
|---|---|---|
| `LookM` | [Paper](https://aclanthology.org/2024.findings-emnlp.235/) · [Code](https://github.com/SUSTechBruce/LOOK-M) (Findings of EMNLP 2024) | 基于文本先验驱逐视觉 KV，并合并被驱逐的视觉 KV |
| `MEDA` | [Paper](https://aclanthology.org/2025.naacl-long.125/) · [Code](https://github.com/aiot-mlsys-lab/meda) (NAACL 2025) | 基于多模态注意力熵做分层动态预算 |
| `FastV` | [Paper](https://arxiv.org/abs/2403.06764) · [Code](https://github.com/pkunlp-icler/FastV) (ECCV 2024 Oral) | 在早期 filter layer 之后剪枝视觉 token |
| `FitPrune` | [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/34366) · [Code](https://github.com/ywh187/FitPrune) (AAAI 2025) | 按层调度的渐进式视觉 token 剪枝 |
| `SparseMM` | [Paper](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SparseMM_Head_Sparsity_Emerges_from_Visual_Concept_Responses_in_MLLMs_ICCV_2025_paper.html) · [Code](https://github.com/CR400AF-A/SparseMM) (ICCV 2025) | 基于视觉 head 重要性的 head-wise 预算 |
| `MixKV` | [Paper](https://arxiv.org/abs/2510.20707) · [Code](https://github.com/xuyang-liu16/MixKV) (ICLR 2026 Poster) | 按 head 混合重要性和多样性分数 |
| `GUIKV` | [Paper](https://arxiv.org/abs/2510.00536) · [Code](https://github.com/SalesforceAIResearch/GUI-KV) | 面向 GUI agent 的空间显著性和时间冗余建模 |
| `STaRKV` | [Paper](https://arxiv.org/abs/2606.01790) · [Code](https://github.com/kawhiiiileo/STaR-KV) | 空间互信息先验和熵引导的分数锐化 |
| `InfiniPotV` | [Paper](https://openreview.net/forum?id=hFxOZjHyTg) · [Code](https://github.com/aiha-lab/InfiniPot-V) (NeurIPS 2025 Poster) | 只驱逐视觉 KV，结合时间冗余和值范数，面向流式视频 |
| `VidKV` | [Paper](https://arxiv.org/abs/2503.16257) · [Code](https://github.com/KD-TAO/VidKV) | 视觉 token 的约 1.x-bit 混合精度 KV 量化 |

与官方实现的差异，以及相关测量细节，写在各方法的 docstring 中。

## 快速开始

用 `compress(model, method)` 包住任意 forward 或 `generate` 调用：

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

开启 `verbose=True` 后，推理结束会打印实际测得的 KV cache 大小。这里统计的是实际 cache，
不是直接用请求的 `ratio` 推导出来的数值，因为文本 token、recent window 等通常会被保护：

```text
[mllm_kvcompress] LookM compression summary (28 layers)
  KV cache before: 281 tokens/layer, 7.68 MiB
  KV cache after:  140 tokens/layer, 3.83 MiB
  measured compression ratio: 50.2%
```

对 head-wise budget 方法，物理 cache 大小可能保持不变，方法会在 decoding 时屏蔽被驱逐的 key；
这类方法的 summary 会额外报告测得的 virtual compression ratio。

本地 checkpoint 也可以直接传入，例如 `model_weight/Qwen3-VL-2B-Thinking` 或
`model_weight/Qwen3.5-0.8B`，前提是当前 transformers 版本包含对应架构。

也可以通过 registry 按名称创建方法：

```python
from mllm_kvcompress import create_method, METHODS

print(sorted(METHODS))  # ['fastv', 'fitprune', 'gui_kv', ...]
method = create_method("sparsemm", ratio=0.5)
```

## 进阶用法

### 多问题共享一个压缩后的上下文

评测时，问题本身不应该被作为上下文一起压缩掉。压缩后的 cache 应该可以服务任意后续问题。
内置 pipeline 会先 pre-fill 并压缩多模态上下文，然后在压缩 cache 上回答每个问题：

```python
import requests
from PIL import Image
from transformers import pipeline
import mllm_kvcompress  # 注册 pipeline
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
    verbose=True,
)
print(result["answers"])
```

## Benchmark

```bash
pip install -e ".[bench]"
```

Benchmark 接口是框架级能力，不依赖任何特定压缩方法。当前接入的 native benchmark 包括
`nocaps`、`flickr30k`、`aokvqa`、`mmmu`、`pca_bench`、`ocr_vqa`；
后续会逐步增加更多 benchmark。

选择模型、压缩策略和测试数据集后即可运行：

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --settings baseline,look_m_0.5,fastv_0.5 \
  --benchmarks nocaps,aokvqa \
  --limit 100 \
  --output-dir runs/qwen2vl_native
```

`--settings all` 会运行 full-cache baseline，以及所有已注册压缩策略的标准 50% 配置
（`VidKV` 使用自己的默认 bit-width）。默认数据源来自 Hugging Face：
`HuggingFaceM4/NoCaps`、`lmms-lab/flickr30k`、`HuggingFaceM4/A-OKVQA`、`MMMU/MMMU`、
`PCA-Bench/PCA-Bench-V1`、`howard-hou/OCR-VQA`。

也可以把某个数据集替换成本地路径或其他 Hugging Face id：

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model /path/to/model \
  --settings baseline,look_m_0.5 \
  --benchmarks aokvqa,ocr_vqa \
  --source aokvqa=/data/hkustgz/datasets/A-OKVQA \
  --source ocr_vqa=/path/to/native/ocr-vqa.json \
  --output-dir runs/local_sources
```

正式加载模型前，可以先检查数据字段和第一条样本的媒体加载是否正常：

```bash
python -m mllm_kvcompress.benchmarks.native_runner \
  --model /path/to/model \
  --benchmarks nocaps \
  --check-data-only \
  --limit 1
```

同样的能力也可以通过 Python API 调用：

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

benchmark registry 的组织方式和 method registry 对齐：每个数据集放在
`mllm_kvcompress/benchmarks/<benchmark>/` 下，每个压缩策略从 `mllm_kvcompress.methods` 选择。

PCA-Bench 公开的 `test_closed` split 可能隐藏标签（`answer_index=-1`）。这种情况下仍会生成预测，
评测结果会报告 unlabeled 数量；如果需要分数，请提供带标签的本地数据源。

## 致谢

本项目受到 [NVIDIA kvpress](https://github.com/NVIDIA/kvpress) 启发。kvpress 提出了
hook-based、one-file-per-method 的文本 LLM KV cache 压缩组织方式。

感谢以下开源 MLLM KV cache 压缩方法的作者：
[LOOK-M](https://github.com/SUSTechBruce/LOOK-M)、[MEDA](https://github.com/AIoT-MLSys-Lab/MEDA)、
[FastV](https://github.com/pkunlp-icler/fastv)、[FitPrune](https://github.com/ywh187/FitPrune)、
[SparseMM](https://github.com/CR400AF-A/SparseMM)、[MixKV](https://github.com/xuyang-liu16/MixKV)、
[GUI-KV](https://github.com/SalesforceAIResearch/GUI-KV)、[STaR-KV](https://github.com/kawhiiiileo/STaR-KV)、
[InfiniPot-V](https://github.com/aiha-lab/InfiniPot-V) 和 [VidKV](https://github.com/KD-TAO/VidKV)。

## 引用

如果本项目对你的工作有帮助，请引用：

```bibtex
@software{mllm_kvcompress,
  author = {Chen, Hong},
  title  = {MLLM-KVCompress: KV cache compression for multimodal LLMs},
  url    = {https://github.com/ChenHong30/MLLM-KVCompress},
  year   = {2026}
}
```

也请引用你使用的压缩方法原论文，见上方[方法表](#支持压缩方法)。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ChenHong30/MLLM-KVCompress&type=Date)](https://star-history.com/#ChenHong30/MLLM-KVCompress&Date)
