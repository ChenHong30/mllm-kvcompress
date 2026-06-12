# SPDX-License-Identifier: Apache-2.0

from mllm_kvcompress.methods.fastv import FastV
from mllm_kvcompress.methods.fitprune import FitPrune
from mllm_kvcompress.methods.gui_kv import GUIKV
from mllm_kvcompress.methods.infinipot_v import InfiniPotV
from mllm_kvcompress.methods.look_m import LookM
from mllm_kvcompress.methods.meda import MEDA
from mllm_kvcompress.methods.mixkv import MixKV
from mllm_kvcompress.methods.sparsemm import SparseMM
from mllm_kvcompress.methods.star_kv import STaRKV
from mllm_kvcompress.methods.vidkv import VidKV

# Registry of all available compression methods, keyed by a short name
METHODS = {
    "look_m": LookM,
    "meda": MEDA,
    "fastv": FastV,
    "fitprune": FitPrune,
    "sparsemm": SparseMM,
    "mixkv": MixKV,
    "gui_kv": GUIKV,
    "star_kv": STaRKV,
    "infinipot_v": InfiniPotV,
    "vidkv": VidKV,
}


def create_method(name: str, **kwargs):
    """Instantiate a compression method by its registry name, e.g. create_method("look_m", ratio=0.5)."""
    if name not in METHODS:
        raise KeyError(f"Unknown method '{name}', available: {sorted(METHODS)}")
    return METHODS[name](**kwargs)


__all__ = [
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
]
