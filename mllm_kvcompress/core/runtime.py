# SPDX-License-Identifier: Apache-2.0
"""
Runtime that connects compression methods to a multimodal model.

A `CompressionMethod` describes *what* to keep; the runtime decides *when*: it
captures modality information before the backbone runs, intercepts every attention
layer at the end of pre-filling, builds a `LayerContext` and writes the compressed
KV pairs back into the cache. Compression never runs during decoding.

Usage
-----
>>> from mllm_kvcompress import compress, LookM
>>> with compress(model, LookM(ratio=0.5)):
...     model.generate(**inputs)
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Callable, Generator, Optional

import torch
from torch import nn
from transformers import PreTrainedModel

from mllm_kvcompress.core.cache import (
    language_model_of,
    multimodal_backbone_of,
    per_layer_attention_masks,
    read_layer_kv,
    write_layer_kv,
)
from mllm_kvcompress.core.modality import vision_mask_from_input_ids
from mllm_kvcompress.core.state import LayerContext

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = (
    "Qwen2VLForConditionalGeneration",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
    "Qwen3_5ForConditionalGeneration",
    "InternVLChatModel",
)


class _PatchHandle:
    """Small removable handle for temporary monkey patches."""

    def __init__(self, remove: Callable[[], None]):
        self._remove = remove

    def remove(self):
        self._remove()


@dataclass
class CompressionMethod:
    """
    Base class for all multimodal KV cache compression methods.

    Subclasses implement `compress(ctx)`, receiving a `LayerContext` with the layer's
    KV pairs, the vision token mask and attention helpers, and returning the
    compressed (keys, values). Methods that need information across layers (e.g.
    layer-wise budget allocation) can defer work to `post_prefill_hooks`.
    """

    def setup(self, model: PreTrainedModel):
        """Optional initialization from the model before hooks are installed."""
        pass

    @property
    def vision_mask(self) -> Optional[torch.Tensor]:
        """Vision token mask (batch_size, seq_len) of the current pre-fill, or None."""
        return getattr(self, "_vision_mask", None)

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress one layer's KV pairs at the end of pre-filling."""
        raise NotImplementedError("compress must be implemented in subclass")

    def post_prefill_hooks(self, backbone: nn.Module) -> list:
        """
        Optional extension point: extra hooks for methods that act once pre-filling is
        complete (e.g. global layer-budget allocation).
        """
        return []

    def stats_note(self) -> Optional[str]:
        """Optional extra line for the verbose compression summary."""
        return None

    # ------------------------------------------------------------------
    # Runtime internals
    # ------------------------------------------------------------------

    def _capture_modality_from_input_ids(self, input_ids: Optional[torch.Tensor], module: nn.Module):
        """Capture the vision token mask from input ids when they are available."""
        if input_ids is None or input_ids.dim() != 2 or input_ids.shape[1] <= 1:
            return
        mask = vision_mask_from_input_ids(input_ids, module.config, module)
        if mask is None:
            logger.warning("No image/video token ids in the model config, vision_mask set to None")
        self._vision_mask = mask

    def _capture_modality(self, module: nn.Module, args: tuple, kwargs: dict):
        """Forward pre-hook on the multimodal backbone: capture the vision token mask."""
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        self._capture_modality_from_input_ids(input_ids, module)

    def _patch_generate_capture(self, model: PreTrainedModel):
        """
        Capture modality before custom generate methods that bypass the multimodal
        forward path. InternVLChatModel.generate builds `inputs_embeds` and calls the
        inner language model directly, so a normal forward pre-hook never sees ids.
        """
        if not hasattr(model, "generate"):
            return None

        original = model.generate

        def wrapped_generate(instance, *args, **kwargs):
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) >= 2:
                input_ids = args[1]
            self._capture_modality_from_input_ids(input_ids, instance)
            return original(*args, **kwargs)

        model.generate = MethodType(wrapped_generate, model)
        return _PatchHandle(lambda: setattr(model, "generate", original))

    def _layer_hook(self, module: nn.Module, args: list, kwargs: dict, output: list):
        """Forward hook on each attention layer: compress the cache during pre-filling."""
        # kwarg renamed from past_key_value to past_key_values across transformers versions
        cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
        if cache is None:
            return output

        hidden_states = kwargs["hidden_states"]
        keys, values = read_layer_kv(cache, module.layer_idx)

        # Only compress during pre-filling: from an empty cache, the cache length
        # after the update equals the query length
        if keys.shape[2] != hidden_states.shape[1]:
            # First decoding step: the cache holds its final compressed length plus the
            # new query tokens, measured here so that deferred (post-prefill) methods
            # are also accounted for
            stats = self._stats
            if stats is not None and module.layer_idx in stats["before_tokens"]:
                if module.layer_idx not in stats["after_tokens"]:
                    stats["after_tokens"][module.layer_idx] = keys.shape[2] - hidden_states.shape[1]
            return output

        vision_mask = self._vision_mask
        if vision_mask is not None and vision_mask.shape[1] != keys.shape[2]:
            logger.warning(
                "vision_mask length (%d) does not match cache length (%d), ignoring it. "
                "Note that chunked pre-filling is not supported.",
                vision_mask.shape[1],
                keys.shape[2],
            )
            self._vision_mask = vision_mask = None

        stats = self._stats
        if stats is not None:
            stats["before_tokens"][module.layer_idx] = keys.shape[2]
            stats["bytes_per_token"][module.layer_idx] = (
                keys.numel() + values.numel()
            ) * keys.element_size() / keys.shape[2]
            stats["cache"] = cache
            stats["after_tokens"].pop(module.layer_idx, None)

        ctx = LayerContext(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=output[1] if isinstance(output, (tuple, list)) and len(output) > 1 else None,
            vision_mask=vision_mask,
            kwargs=kwargs,
        )
        keys, values = self.compress(ctx)
        write_layer_kv(cache, module.layer_idx, keys, values)

        if stats is not None:
            evicted = getattr(module, "evicted_key_indices", None)
            if evicted is not None:
                stats["virtual_evicted"][module.layer_idx] = int(evicted[0].numel())
                stats["virtual_total"][module.layer_idx] = keys.shape[0] * keys.shape[1] * keys.shape[2]
        return output

    def install(self, model: PreTrainedModel) -> list:
        """Register all hooks on the model and return their handles."""
        if model.__class__.__name__ not in SUPPORTED_MODELS:
            logger.warning(f"Model {model.__class__.__name__} not tested, supported models: {SUPPORTED_MODELS}")

        self.setup(model)
        self._vision_mask = None
        self._stats = {
            "before_tokens": {},
            "bytes_per_token": {},
            "after_tokens": {},
            "virtual_evicted": {},
            "virtual_total": {},
            "cache": None,
        }

        backbone = multimodal_backbone_of(model)
        language_model = language_model_of(model)

        hooks = [backbone.register_forward_pre_hook(self._capture_modality, with_kwargs=True)]
        generate_patch = self._patch_generate_capture(model)
        if generate_patch is not None:
            hooks.append(generate_patch)
        for layer in language_model.layers:
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                hooks.append(attention.register_forward_hook(self._layer_hook, with_kwargs=True))

        post_modules = []
        for module in (backbone, getattr(backbone, "language_model", None), getattr(model, "language_model", None)):
            if module is not None and all(id(module) != id(existing) for existing in post_modules):
                post_modules.append(module)
        for module in post_modules:
            hooks.extend(self.post_prefill_hooks(module))
        return hooks

    def uninstall(self, hooks: list):
        for hook in hooks:
            hook.remove()
        self._vision_mask = None
        # finalize the measured sizes and release the cache reference
        stats = getattr(self, "_stats", None)
        if stats and stats["cache"] is not None:
            for layer_idx in stats["before_tokens"]:
                if layer_idx not in stats["after_tokens"]:
                    stats["after_tokens"][layer_idx] = read_layer_kv(stats["cache"], layer_idx)[0].shape[2]
            stats["cache"] = None

    def compression_summary(self) -> str:
        """
        Human-readable summary of the last pre-fill, with measured (not nominal) sizes.

        Sizes after compression are measured on the first decoding step, so that
        methods compressing after pre-filling (layer-wise budgets) are also reported
        correctly; without decoding they are read from the cache directly.
        """
        stats = getattr(self, "_stats", None)
        if not stats or not stats["before_tokens"]:
            return f"[mllm_kvcompress] {self.__class__.__name__}: no pre-fill was captured, nothing to report"

        layer_indices = sorted(stats["before_tokens"])
        for layer_idx in layer_indices:
            if layer_idx not in stats["after_tokens"] and stats["cache"] is not None:
                stats["after_tokens"][layer_idx] = read_layer_kv(stats["cache"], layer_idx)[0].shape[2]

        before_tokens = [stats["before_tokens"][i] for i in layer_indices]
        after_tokens = [stats["after_tokens"].get(i, stats["before_tokens"][i]) for i in layer_indices]
        before_bytes = sum(stats["before_tokens"][i] * stats["bytes_per_token"][i] for i in layer_indices)
        after_bytes = sum(stats["after_tokens"].get(i, stats["before_tokens"][i]) * stats["bytes_per_token"][i] for i in layer_indices)

        def fmt_tokens(tokens: list[int]) -> str:
            return str(tokens[0]) if min(tokens) == max(tokens) else f"{min(tokens)}-{max(tokens)} (mean {sum(tokens) / len(tokens):.0f})"

        lines = [
            f"[mllm_kvcompress] {self.__class__.__name__} compression summary ({len(layer_indices)} layers)",
            f"  KV cache before: {fmt_tokens(before_tokens)} tokens/layer, {before_bytes / 2**20:.2f} MiB",
            f"  KV cache after:  {fmt_tokens(after_tokens)} tokens/layer, {after_bytes / 2**20:.2f} MiB",
            f"  measured compression ratio: {1 - after_bytes / before_bytes:.1%}",
        ]
        if stats["virtual_evicted"]:
            evicted = sum(stats["virtual_evicted"].values())
            total = sum(stats["virtual_total"].values())
            lines.append(
                f"  virtual compression ratio: {evicted / total:.1%} "
                "(per-head evicted keys are neutralized during decoding; memory is not reduced)"
            )
        note = self.stats_note()
        if note:
            lines.append(f"  {note}")
        return "\n".join(lines)


@contextmanager
def compress(model: PreTrainedModel, method: Optional[CompressionMethod], verbose: bool = False) -> Generator:
    """
    Apply a KV cache compression method to a multimodal model within a context.

    Hooks are installed on entry and removed on exit. Passing method=None is a no-op,
    which is convenient for baselines. With verbose=True, a summary with the measured
    KV cache sizes before/after compression is printed on exit.
    """
    if method is None:
        yield
        return
    hooks = method.install(model)
    try:
        # methods can compress layers to different lengths, which requires aligning
        # the shared causal mask to each layer's KV length during decoding
        with per_layer_attention_masks(model):
            yield
        if verbose:
            print(method.compression_summary())
    finally:
        method.uninstall(hooks)
