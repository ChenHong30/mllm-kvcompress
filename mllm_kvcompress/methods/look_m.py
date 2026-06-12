# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext
from mllm_kvcompress.methods.merging import text_prior_select_and_merge

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

    Adaptation notes: the official implementation accumulates attention over the full
    eager pre-fill attention map. To stay compatible with sdpa/flash attention, the
    heavy hitter scores are computed from an observation window of recent queries.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=32
        Number of recent queries used to compute attention-based importance scores.
    recent_fraction : float, default=0.5
        Fraction of the kept budget reserved for the most recent tokens (the official
        implementation uses an equal split between heavy hitters and recent tokens).
    merge : str, default="pivot"
        Merge strategy for evicted KV pairs: "pivot", "average", "weighted" or "none".
        The flagship LOOK-M variant is text-prior + pivot merge.
    merge_keys : bool, default=False
        Whether to also merge keys (official LOOK-M behavior on LLaVA). Disabled by
        default because merging post-RoPE keys corrupts generation under the mrope
        position encoding of Qwen2-VL (see methods.merging).
    """

    ratio: float = 0.0
    window_size: int = 32
    recent_fraction: float = 0.5
    merge: str = "pivot"
    merge_keys: bool = False

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert self.merge in ("pivot", "average", "weighted", "none"), f"Unknown merge strategy: {self.merge}"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values

        if ctx.vision_mask is None:
            logger.warning_once(
                "LookM could not find vision token positions, falling back to "
                "modality-agnostic eviction (no text-prior)."
            )

        # Accumulated attention received by each position (H2O-style heavy hitter score)
        scores = ctx.to_kv_heads(ctx.window_attention(self.window_size).sum(dim=-2))
        n_kept = int(ctx.seq_len * (1 - self.ratio))
        return text_prior_select_and_merge(
            ctx.keys, ctx.values, scores, ctx.vision_mask, n_kept, self.recent_fraction, self.merge, self.merge_keys
        )
