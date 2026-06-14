# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

import torch
from torch import nn

from mllm_kvcompress.core.cache import read_layer_kv, write_layer_kv
from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext
from mllm_kvcompress.methods.merging import text_prior_select_and_merge


@dataclass
class MEDA(CompressionMethod):
    """
    MEDA: dynamic layer-wise KV cache allocation guided by multimodal attention entropy.

    MEDA observes that the complexity of cross-modal interaction varies across layers
    and allocates the KV cache budget accordingly: layers with a higher multimodal
    attention entropy receive a larger share. Within each layer, KV pairs are selected
    with the text-prior heavy-hitter scheme of LOOK-M and evicted KV pairs are merged
    into the kept ones.

    Because the allocation requires the entropies of *all* layers, compression is
    deferred: the per-layer step only records entropy and importance scores, and a
    post-prefill hook on the multimodal backbone performs the global allocation and
    compresses every layer once pre-filling is complete.

    Based on MEDA (https://arxiv.org/abs/2502.17599, NAACL 2025),
    official implementation: https://github.com/AIoT-MLSys-Lab/MEDA.

    Adaptation notes: the official implementation computes the entropy and heavy-hitter
    scores on the full eager pre-fill attention map and compresses at the first decoding
    step. The full map does not fit in memory for the long contexts of recent multimodal
    models, so both are accumulated in query-chunks (the exact same all-query quantities;
    see `LayerContext.accumulated_attention_and_entropy`).

    Parameters
    ----------
    ratio : float, default=0.0
        Average fraction of key-value pairs to remove across layers.
    score_chunk_size : int, default=512
        Query-chunk size used to accumulate the entropy and importance scores from the
        full pre-fill attention without materializing the (seq_len, seq_len) map.
    recent_fraction : float, default=0.5
        Fraction of each layer's kept budget reserved for the most recent tokens
        (the official implementation uses an equal split).
    merge : str, default="average"
        Merge strategy for evicted KV pairs: "pivot", "average", "weighted" or "none".
        The official MEDA flagship uses text-prior average merging.
    min_keep_ratio : float, default=0.01
        Lower clamp for the per-layer keep ratio (matches the official implementation).
    merge_keys : bool, default=False
        Whether to also merge keys (official behavior on LLaVA). Disabled by default,
        see methods.merging.
    """

    ratio: float = 0.0
    score_chunk_size: int = 512
    recent_fraction: float = 0.5
    merge: str = "average"
    min_keep_ratio: float = 0.01
    merge_keys: bool = False

    _entropies: dict = field(default_factory=dict, init=False, repr=False)
    _scores: dict = field(default_factory=dict, init=False, repr=False)
    _cache: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert self.merge in ("pivot", "average", "weighted", "none"), f"Unknown merge strategy: {self.merge}"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        """Record this layer's entropy and scores; actual compression is deferred."""
        if self.ratio == 0:
            return ctx.keys, ctx.values

        scores, entropy = ctx.accumulated_attention_and_entropy(self.score_chunk_size)

        # Multimodal attention entropy, head-mean normalized by the number of queries
        self._entropies[ctx.layer_idx] = entropy[0].mean() / ctx.seq_len
        self._scores[ctx.layer_idx] = ctx.to_kv_heads(scores)
        self._cache = ctx.kwargs.get("past_key_values", ctx.kwargs.get("past_key_value"))

        return ctx.keys, ctx.values

    def _allocate_and_compress(self, module: nn.Module, args, kwargs, output):
        """
        Post-prefill hook on the multimodal backbone: allocate per-layer keep ratios
        from the recorded entropies and compress every layer.
        """
        if not self._scores:
            return output

        layer_indices = sorted(self._scores.keys())
        num_layers = len(layer_indices)
        entropies = torch.stack([self._entropies[idx] for idx in layer_indices])

        # Budget allocation: softmax over layer entropies, scaled to the global budget
        weights = torch.softmax(entropies, dim=0)
        layer_keep_ratios = (weights * num_layers * (1 - self.ratio)).clamp(min=self.min_keep_ratio, max=1.0)

        for i, layer_idx in enumerate(layer_indices):
            keys, values = read_layer_kv(self._cache, layer_idx)
            n_kept = int(keys.shape[2] * layer_keep_ratios[i].item())
            keys, values = text_prior_select_and_merge(
                keys,
                values,
                self._scores[layer_idx],
                self.vision_mask,
                n_kept,
                self.recent_fraction,
                self.merge,
                self.merge_keys,
            )
            write_layer_kv(self._cache, layer_idx, keys, values)

        self._entropies.clear()
        self._scores.clear()
        self._cache = None
        return output

    def post_prefill_hooks(self, backbone: nn.Module) -> list:
        self._entropies.clear()
        self._scores.clear()
        self._cache = None
        return [backbone.register_forward_hook(self._allocate_and_compress, with_kwargs=True)]
