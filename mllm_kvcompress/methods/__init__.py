# SPDX-License-Identifier: Apache-2.0

from typing import Any

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


def parse_settings(settings: str, default: list[str] | None = None):
    if not settings or settings == "all":
        names = default or ["baseline", *sorted(METHODS)]
    else:
        names = [item.strip() for item in settings.replace(";", ",").split(",") if item.strip()]

    return [parse_setting(name) for name in names]


def parse_setting(setting: str):
    if setting == "baseline":
        return "baseline", lambda: None

    method_name, kwargs = parse_method_setting(setting)
    if method_name not in METHODS:
        raise ValueError(f"Unknown compression method '{method_name}'. Available: {['baseline', *sorted(METHODS)]}")

    def factory():
        return METHODS[method_name](**kwargs)

    suffix = "_".join(f"{key}{value}" for key, value in kwargs.items())
    setting_name = method_name if not suffix else f"{method_name}_{suffix}"
    return setting_name, factory


def parse_method_setting(setting: str) -> tuple[str, dict[str, Any]]:
    if ":" in setting:
        method_name, raw_kwargs = setting.split(":", 1)
        kwargs = {}
        for item in raw_kwargs.replace("|", ";").split(";"):
            item = item.strip()
            if not item:
                continue
            key, value = item.split("=", 1)
            kwargs[key.strip()] = _parse_value(value.strip())
        return method_name.replace("-", "_"), kwargs

    normalized = setting.replace("-", "_")
    for method_name in sorted(METHODS, key=len, reverse=True):
        prefix = f"{method_name}_"
        if normalized.startswith(prefix):
            tail = normalized[len(prefix) :]
            try:
                return method_name, {"ratio": float(tail)}
            except ValueError:
                pass
    return normalized, {}


def format_method_setting(method: str, method_kwargs: dict[str, Any] | None = None) -> str:
    method = method.replace("-", "_")
    if not method_kwargs:
        return method
    raw_kwargs = ";".join(f"{key}={value}" for key, value in method_kwargs.items())
    return f"{method}:{raw_kwargs}"


def _parse_value(value: str):
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


__all__ = [
    "METHODS",
    "create_method",
    "format_method_setting",
    "parse_method_setting",
    "parse_setting",
    "parse_settings",
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
