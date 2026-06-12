# SPDX-License-Identifier: Apache-2.0
"""KV cache and model-structure helpers, compatible with transformers 4.x and 5.x."""

from collections.abc import Mapping
from contextlib import contextmanager

import torch
from torch import nn
from transformers import Cache


def read_layer_kv(cache: Cache, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read keys and values of one cache layer.
    Handles both the post-4.54 `cache.layers` API and the legacy `key_cache`/`value_cache` lists.
    """
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        if not (hasattr(layer, "keys") and hasattr(layer, "values")):
            raise ValueError(f"Cache layer {layer_idx} does not store key/value tensors")
        if layer.keys is None or layer.values is None:
            raise ValueError(f"Cache layer {layer_idx} has no key/value tensors yet")
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def write_layer_kv(cache: Cache, layer_idx: int, keys: torch.Tensor, values: torch.Tensor):
    """Write (compressed) keys and values back into one cache layer in-place."""
    if hasattr(cache, "layers"):
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
    else:
        cache.key_cache[layer_idx] = keys
        cache.value_cache[layer_idx] = values


def has_layer_kv(cache: Cache, layer_idx: int) -> bool:
    """Return whether a cache layer stores attention key/value tensors."""
    if hasattr(cache, "layers"):
        if layer_idx >= len(cache.layers):
            return False
        layer = cache.layers[layer_idx]
        return hasattr(layer, "keys") and hasattr(layer, "values") and layer.keys is not None
    return layer_idx < len(cache.key_cache) and cache.key_cache[layer_idx] is not None


def _clone_state(state):
    if isinstance(state, torch.Tensor):
        return state.clone()
    if isinstance(state, list):
        return [_clone_state(item) for item in state]
    if isinstance(state, tuple):
        return tuple(_clone_state(item) for item in state)
    if isinstance(state, dict):
        return {key: _clone_state(value) for key, value in state.items()}
    return state


def snapshot_cache(cache: Cache, layer_indices: list[int] | None = None) -> dict:
    """
    Snapshot mutable cache state before decoding a branch.

    KV layers are represented by their original sequence lengths to avoid copying the
    full cache. Linear-attention recurrent states (used by Qwen3.5) are cloned because
    they are updated in-place during decoding and cannot be rewound by truncation.
    """
    if layer_indices is None:
        layer_indices = list(range(len(cache)))

    snapshot = {"seq_lengths": {}, "states": {}}
    for layer_idx in layer_indices:
        if has_layer_kv(cache, layer_idx):
            keys, _ = read_layer_kv(cache, layer_idx)
            snapshot["seq_lengths"][layer_idx] = keys.shape[2]

    if hasattr(cache, "layers"):
        for layer_idx, layer in enumerate(cache.layers):
            states = {}
            for name in ("conv_states", "recurrent_states"):
                if hasattr(layer, name):
                    value = getattr(layer, name)
                    if value is not None:
                        states[name] = _clone_state(value)
            if states:
                snapshot["states"][layer_idx] = states

    return snapshot


def rewind_cache(cache: Cache, seq_lengths: Mapping[int, int] | list[int]):
    """Truncate each cache layer back to the given lengths (e.g. to drop a generated answer)."""
    items = seq_lengths.items() if isinstance(seq_lengths, Mapping) else enumerate(seq_lengths)
    for layer_idx, seq_length in items:
        if not has_layer_kv(cache, layer_idx):
            continue
        keys, values = read_layer_kv(cache, layer_idx)
        write_layer_kv(cache, layer_idx, keys[:, :, :seq_length], values[:, :, :seq_length])


def restore_cache(cache: Cache, snapshot: dict):
    """Restore a snapshot created by `snapshot_cache`."""
    rewind_cache(cache, snapshot.get("seq_lengths", {}))
    if hasattr(cache, "layers"):
        for layer_idx, states in snapshot.get("states", {}).items():
            layer = cache.layers[layer_idx]
            for name, value in states.items():
                setattr(layer, name, _clone_state(value))


def multimodal_backbone_of(model: nn.Module) -> nn.Module:
    """Return the module that receives multimodal inputs before dispatching to the LLM."""
    base_model = getattr(model, "model", None)
    if base_model is not None and hasattr(base_model, "language_model"):
        return base_model
    return model


def _decoder_layers_of(module: nn.Module):
    if hasattr(module, "layers"):
        return module
    nested = getattr(module, "model", None)
    if nested is not None and hasattr(nested, "layers"):
        return nested
    return None


def language_model_of(model: nn.Module) -> nn.Module:
    """Return the language model holding the decoder layers of a multimodal model."""
    backbone = multimodal_backbone_of(model)
    candidates = []
    for owner in (backbone, model):
        candidates.extend(getattr(owner, name, None) for name in ("language_model", "llm"))
        candidates.append(owner)

    for candidate in candidates:
        if candidate is None:
            continue
        language_model = _decoder_layers_of(candidate)
        if language_model is not None:
            return language_model
    raise ValueError(f"Could not find decoder layers for {model.__class__.__name__}")


def kv_cache_layer_indices(model: nn.Module) -> list[int]:
    """Return cache layer indices for decoder layers that have attention KV tensors."""
    indices = []
    for fallback_idx, layer in enumerate(language_model_of(model).layers):
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            indices.append(getattr(attention, "layer_idx", fallback_idx))
    return indices


@contextmanager
def per_layer_attention_masks(model: nn.Module):
    """
    Context manager slicing the causal attention mask to each layer's actual KV length.

    Compression methods can leave caches whose length differs across layers (layer-wise
    budgets, progressive pruning). The model builds a single causal mask sized for one
    cache length, which makes sdpa fail on layers with a different cache length.
    transformers 4.x sliced the mask inside every attention layer; this was removed in
    v5, so it is restored here with forward pre-hooks. The mask structure is "past fully
    visible + causal block over the new queries", hence keeping the last
    past_len + q_len columns (or left-padding with visible) is exact.
    """
    hooks = []

    def hook(module, args, kwargs):
        mask = kwargs.get("attention_mask")
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)
        if not isinstance(mask, torch.Tensor) or cache is None or hidden_states is None:
            return
        if module.layer_idx >= len(cache):
            return
        if not has_layer_kv(cache, module.layer_idx):
            return
        kv_length = read_layer_kv(cache, module.layer_idx)[0].shape[2] + hidden_states.shape[1]
        if mask.shape[-1] > kv_length:
            kwargs["attention_mask"] = mask[..., -kv_length:]
            return args, kwargs
        if mask.shape[-1] < kv_length:
            pad_value = True if mask.dtype == torch.bool else 0.0
            pad = torch.full(
                (*mask.shape[:-1], kv_length - mask.shape[-1]), pad_value, dtype=mask.dtype, device=mask.device
            )
            kwargs["attention_mask"] = torch.cat([pad, mask], dim=-1)
            return args, kwargs

    try:
        for layer in language_model_of(model).layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                hooks.append(attention.register_forward_pre_hook(hook, with_kwargs=True))
        yield
    finally:
        for h in hooks:
            h.remove()
