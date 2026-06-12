# SPDX-License-Identifier: Apache-2.0
"""
mllm-kvcompress: KV cache compression for multimodal LLMs.

Quick start
-----------
>>> from mllm_kvcompress import compress, LookM
>>> with compress(model, LookM(ratio=0.5)):
...     model.generate(**inputs)
"""

from mllm_kvcompress.core import CompressionMethod, LayerContext, ScoredEviction, compress
from mllm_kvcompress.core.attention import enable_headwise_masking
from mllm_kvcompress.generation import CompressedContextPipeline
from mllm_kvcompress.methods import (
    METHODS,
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
    create_method,
)

__version__ = "0.2.0"

# enable head-wise (per-head budget) methods such as SparseMM
enable_headwise_masking()

__all__ = [
    "compress",
    "CompressionMethod",
    "ScoredEviction",
    "LayerContext",
    "METHODS",
    "create_method",
    "LookM",
    "MEDA",
    "FastV",
    "FitPrune",
    "SparseMM",
    "MixKV",
    "GUIKV",
    "STaRKV",
    "InfiniPotV",
    "VidKV",
    "CompressedContextPipeline",
]
