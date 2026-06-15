# SPDX-License-Identifier: Apache-2.0
"""Model adapters shared by benchmark implementations."""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

from mllm_kvcompress import CompressionMethod, compress

IMAGE_PLACEHOLDER = "<ImageHere>"


def _torch_dtype(name: str | None):
    if name in (None, "auto"):
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype '{name}', expected auto, bf16, fp16, or fp32")


def _from_pretrained(cls, model_name_or_path: str, **kwargs):
    """Call from_pretrained across transformers versions that renamed torch_dtype."""
    missing_shards = _missing_checkpoint_shards(model_name_or_path)
    if missing_shards:
        preview = ", ".join(missing_shards[:3])
        if len(missing_shards) > 3:
            preview += f", ... ({len(missing_shards)} missing)"
        raise FileNotFoundError(
            f"Checkpoint at {model_name_or_path} is incomplete; missing shard files: {preview}"
        )
    dtype = kwargs.pop("dtype")
    try:
        return cls.from_pretrained(model_name_or_path, dtype=dtype, **kwargs)
    except TypeError:
        if dtype != "auto":
            kwargs["torch_dtype"] = dtype
        return cls.from_pretrained(model_name_or_path, **kwargs)


def _to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if hasattr(value, "to") else value
    return moved


def _model_slug(model_name_or_path: str) -> str:
    name = Path(model_name_or_path).name or model_name_or_path
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _missing_checkpoint_shards(model_name_or_path: str) -> list[str]:
    path = Path(model_name_or_path)
    if not path.is_dir():
        return []
    missing = []
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = path / index_name
        if not index_path.exists():
            continue
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                weight_map = json.load(handle).get("weight_map", {})
        except Exception:
            continue
        for shard in sorted(set(weight_map.values())):
            if not (path / shard).exists():
                missing.append(shard)
    return missing


@dataclass
class BaseModelAdapter:
    """Small interface benchmarks use to run a supported multimodal model."""

    model_name_or_path: str
    model: Any
    processor: Any
    tokenizer: Any
    device: str

    @property
    def model_slug(self) -> str:
        return _model_slug(self.model_name_or_path)

    @property
    def default_max_context_len(self) -> int:
        for owner in (getattr(self.model, "config", None), getattr(getattr(self.model, "config", None), "text_config", None)):
            value = getattr(owner, "max_position_embeddings", None)
            if value:
                return int(value)
        return 4096

    @property
    def default_tokens_per_image(self) -> int:
        return 1024

    def encode_text(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode_tokens(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def load_images(self, image_paths: list[str]) -> list[Image.Image]:
        return [Image.open(path).convert("RGB") for path in image_paths]

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        method_factory: Callable[[], CompressionMethod | None],
        gen_kwargs: dict[str, Any],
        verbose: bool = False,
    ) -> str:
        raise NotImplementedError


class QwenAdapter(BaseModelAdapter):
    """Adapter for processor-based Qwen-style image-text-to-text models.

    Thinking models (e.g. Qwen3-VL) reason before answering. Benchmarks expect a direct
    answer, so the chat template is always rendered with thinking disabled.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str,
        dtype: str | None = "bf16",
        local_files_only: bool = False,
        trust_remote_code: bool = True,
        **_: Any,
    ) -> "QwenAdapter":
        torch_dtype = _torch_dtype(dtype)
        model = _from_pretrained(
            AutoModelForImageTextToText,
            model_name_or_path,
            dtype=torch_dtype,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path, local_files_only=local_files_only, trust_remote_code=trust_remote_code
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        return cls(model_name_or_path, model, processor, tokenizer, device)

    def _messages(self, prompt: str, images: list[Image.Image]) -> list[dict[str, Any]]:
        chunks = prompt.split(IMAGE_PLACEHOLDER)
        content: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks):
            if chunk:
                content.append({"type": "text", "text": chunk})
            if index < len(images):
                content.append({"type": "image", "image": images[index]})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        prompt: str,
        images: list[Image.Image],
        method_factory: Callable[[], CompressionMethod | None],
        gen_kwargs: dict[str, Any],
        verbose: bool = False,
    ) -> str:
        inputs = self.processor.apply_chat_template(
            self._messages(prompt, images),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = _to_device(inputs, self.device)
        input_length = inputs["input_ids"].shape[1]
        with torch.inference_mode(), compress(self.model, method_factory(), verbose=verbose):
            outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.processor.decode(outputs[0, input_length:], skip_special_tokens=True).strip()


def load_model_adapter(
    model_name_or_path: str,
    model_type: str = "auto",
    device: str | None = None,
    dtype: str | None = "bf16",
    local_files_only: bool = False,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> BaseModelAdapter:
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return QwenAdapter.from_pretrained(
        model_name_or_path,
        device,
        dtype=dtype,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
