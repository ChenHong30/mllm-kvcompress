# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.attention import smooth_scores
from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


@dataclass
class SparseMM(CompressionMethod):
    """
    SparseMM: head-wise KV cache budgets driven by visual-head importance.

    SparseMM observes that only a small subset of attention heads ("visual heads")
    actually attend to vision tokens, and allocates the KV cache budget across heads
    proportionally to their visual-head score: visual heads get larger caches, the
    rest fall back to a small uniform budget.

    Based on SparseMM (https://arxiv.org/abs/2506.05344, NeurIPS 2025),
    official implementation: https://github.com/CR400AF-A/SparseMM.

    Adaptation notes:
    - The official pipeline ships *offline* visual-head scores chased on a calibration
      set, normalized across all heads of all layers. Here the visual-head score is
      computed *online* per layer (attention mass each head puts on vision tokens,
      observation-window approximation) and budgets are normalized within the layer.
    - Head-wise budgets produce ragged caches; the official code uses a flattened cache
      with varlen flash attention. Here eviction is *virtual*: evicted keys are
      neutralized during decoding (replaced by keys with zero attention weight, see
      core.attention.headwise_masking). Peak memory is not reduced, but the attention
      sparsity and accuracy of head-wise budgets are reproduced exactly.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to (virtually) remove during compression.
    min_budget_ratio : float, default=0.2
        Fraction of the average per-head budget guaranteed to every head (the `ratio`
        safeguard of the official implementation).
    window_size : int, default=32
        Number of recent queries used to compute attention scores; always kept.
    kernel_size : int, default=7
        Pooling kernel applied to the attention scores.
    pooling : str, default="maxpool"
        Pooling method, "avgpool" or "maxpool" (official default).
    """

    ratio: float = 0.0
    min_budget_ratio: float = 0.2
    window_size: int = 32
    kernel_size: int = 7
    pooling: str = "maxpool"

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert 0 <= self.min_budget_ratio <= 1, "min_budget_ratio must be between 0 and 1"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        ctx.module.evicted_key_indices = None
        if self.ratio == 0:
            return ctx.keys, ctx.values

        window = min(self.window_size, ctx.seq_len - 1)
        attn_weights = ctx.window_attention(window)

        scores = ctx.to_kv_heads(attn_weights.sum(dim=-2))
        scores = smooth_scores(scores, self.kernel_size, self.pooling)
        # always keep the observation window
        scores[..., -window:] = torch.finfo(scores.dtype).max

        # Online visual-head score: attention mass on vision tokens, per kv head
        if ctx.vision_mask is not None:
            vision_attn = attn_weights.mean(dim=-2) * ctx.vision_mask[:, None, :].float()
            head_score = ctx.to_kv_heads(vision_attn).sum(dim=-1)  # (bsz, num_kv_heads)
        else:
            logger.warning_once(
                "SparseMM could not find vision token positions, falling back to uniform head budgets."
            )
            head_score = torch.ones(ctx.batch_size, ctx.num_kv_heads, device=ctx.keys.device)
        head_score = head_score / head_score.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        # Allocate per-head budgets: a guaranteed minimum plus a share of the remainder
        # proportional to the visual-head score (normalized within the layer)
        base_capacity = int(ctx.seq_len * (1 - self.ratio))
        min_budget = int(base_capacity * self.min_budget_ratio)
        remainder = (base_capacity - min_budget) * ctx.num_kv_heads
        budgets = (head_score * remainder + min_budget).round().long().clamp(window, ctx.seq_len)

        # Evict, per head, the (seq_len - budget) lowest-scored positions. The cache
        # tensor is left untouched; evicted positions are neutralized during decoding.
        rank = scores.argsort(dim=-1).argsort(dim=-1)
        evict = rank < (ctx.seq_len - budgets)[..., None]
        ctx.module.evicted_key_indices = evict.nonzero(as_tuple=True)
        return ctx.keys, ctx.values
