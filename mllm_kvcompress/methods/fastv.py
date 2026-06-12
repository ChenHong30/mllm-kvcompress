# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.utils import logging

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


@dataclass
class FastV(CompressionMethod):
    """
    FastV: vision token pruning after an early filtering layer.

    FastV observes that vision tokens receive very little attention after the first
    few layers and discards the least attended vision tokens after layer
    `filter_layer`. This is the representative "indirect" KV compression method: it
    compresses the cache by removing vision tokens entirely rather than by operating
    on already-formed KV statistics.

    Based on FastV (https://arxiv.org/abs/2403.06764, ECCV 2024 Oral),
    official implementation: https://github.com/pkunlp-icler/fastv.

    Adaptation notes: the official implementation drops the tokens from the hidden
    states after `filter_layer`, removing them from all subsequent computation. As a
    cache compression method, we instead drop the KV pairs of the selected vision
    tokens at every layer after `filter_layer` (the pre-fill computation itself is
    unchanged). The attention ranking uses an observation window of recent queries
    instead of the full eager attention map, for sdpa/flash attention compatibility.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of *vision* tokens to remove (the filtering ratio R of the paper).
        Text KV pairs are never removed.
    filter_layer : int, default=2
        Layer whose attention is used to rank vision tokens (K of the paper).
        KV pairs are dropped at all layers strictly after this one.
    window_size : int, default=32
        Number of recent queries used to compute the attention ranking.
    """

    ratio: float = 0.0
    filter_layer: int = 2
    window_size: int = 32

    _kept_positions: Optional[torch.Tensor] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert self.filter_layer >= 0, "filter_layer must be non-negative"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0 or ctx.layer_idx < self.filter_layer:
            return ctx.keys, ctx.values

        if ctx.layer_idx == self.filter_layer:
            self._kept_positions = None
            if ctx.vision_mask is None:
                logger.warning_once("FastV could not find vision token positions, no compression is applied.")
                return ctx.keys, ctx.values

            # Average attention received by each position, averaged over heads and queries
            scores = ctx.window_attention(self.window_size).mean(dim=(1, 2))  # (batch_size, seq_len)

            # Protect text positions, then drop the least attended vision tokens
            n_vision = int(ctx.vision_mask.sum(dim=1).min().item())
            n_dropped = int(n_vision * self.ratio)
            if n_dropped == 0:
                return ctx.keys, ctx.values
            scores = scores.masked_fill(~ctx.vision_mask, torch.finfo(scores.dtype).max)
            n_kept = ctx.seq_len - n_dropped
            self._kept_positions = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values
            return ctx.keys, ctx.values

        if self._kept_positions is None:
            return ctx.keys, ctx.values

        # Layers after filter_layer: drop the KV pairs of the filtered vision tokens
        indices = self._kept_positions[:, None, :, None].expand(-1, ctx.num_kv_heads, -1, ctx.head_dim)
        return ctx.keys.gather(2, indices).contiguous(), ctx.values.gather(2, indices).contiguous()
