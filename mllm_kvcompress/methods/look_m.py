# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext
from mllm_kvcompress.methods.merging import look_m_select_and_merge

logger = logging.get_logger(__name__)


@dataclass
class LookM(CompressionMethod):
    """
    LOOK-M: text-prior KV cache eviction with merging of evicted visual KV pairs.

    LOOK-M observes that text KV pairs matter more than vision KV pairs during decoding
    and (1) keeps a recent window, (2) selects heavy hitters by accumulated attention
    with vision positions penalized so that text KV pairs are evicted last, and
    (3) merges the evicted (mostly visual) KV pairs into their most similar kept KV
    pairs instead of discarding them.

    Based on LOOK-M (https://arxiv.org/abs/2406.18139, Findings of EMNLP 2024),
    official implementation: https://github.com/SUSTechBruce/LOOK-M.

    Parameters
    ----------
    ratio : float, default=0.0
        Approximate fraction of key-value pairs to remove during compression. LOOK-M's
        reference implementation exposes separate heavy-hitter and recent ratios; this
        wrapper derives them as `(1 - ratio) * (1 - recent_fraction)` and
        `(1 - ratio) * recent_fraction`.
    recent_fraction : float, default=0.5
        Fraction of the kept budget reserved for the most recent tokens (the official
        implementation uses an equal split between heavy hitters and recent tokens).
    merge : str, default="pivot"
        Merge strategy for evicted KV pairs: "pivot", "average", "weighted" or "none".
        The flagship LOOK-M variant is text-prior + pivot merge.
    merge_keys : bool, default=True
        Whether to also merge keys. Enabled by default to match the reference LLaVA
        implementation; set it to False for models whose positional encoding makes
        post-RoPE key merging invalid.
    score_chunk_size : int, default=512
        Query-chunk size used to accumulate the heavy-hitter attention scores. The
        reference implementation sums the full eager attention map over all queries,
        which is O(seq_len ** 2) and does not fit in memory for the long contexts of the
        recent multimodal models supported here. The same all-query sum is therefore
        computed in chunks of this many queries, an exact equivalent whose peak memory is
        O(score_chunk_size * seq_len) (see `LayerContext.accumulated_attention`).
    """

    ratio: float = 0.0
    recent_fraction: float = 0.5
    merge: str = "pivot"
    merge_keys: bool = True
    score_chunk_size: int = 512

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert 0 <= self.recent_fraction <= 1, "recent_fraction must be in [0, 1]"
        assert self.merge in ("pivot", "average", "weighted", "none"), f"Unknown merge strategy: {self.merge}"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values

        if ctx.vision_mask is None:
            logger.warning_once(
                "LookM could not find vision token positions, skipping compression for this prefill."
            )
            return ctx.keys, ctx.values

        # The reference implementation sums full prefill attention over all query
        # positions before applying the text-prior mask; accumulated in query-chunks
        # here so the full attention map is never materialized (long-context models).
        scores = ctx.to_kv_heads(ctx.accumulated_attention(self.score_chunk_size))
        keep_ratio = 1.0 - self.ratio
        important_ratio = keep_ratio * (1.0 - self.recent_fraction)
        recent_ratio = keep_ratio * self.recent_fraction
        return look_m_select_and_merge(
            ctx.keys,
            ctx.values,
            scores,
            ctx.vision_mask,
            important_ratio,
            recent_ratio,
            self.merge,
            self.merge_keys,
        )
