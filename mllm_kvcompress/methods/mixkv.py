# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn

from mllm_kvcompress.core.attention import smooth_scores
from mllm_kvcompress.core.scored import ScoredEviction
from mllm_kvcompress.core.state import LayerContext


def _minmax_rescale(scores: torch.Tensor, reference_mean: torch.Tensor) -> torch.Tensor:
    """Min-max normalize per head, then rescale so the mean matches `reference_mean`."""
    mn = scores.amin(dim=-1, keepdim=True)
    mx = scores.amax(dim=-1, keepdim=True)
    scores = (scores - mn) / (mx - mn + 1e-8)
    return scores * reference_mean / (scores.mean(dim=-1, keepdim=True) + 1e-8)


@dataclass
class MixKV(ScoredEviction):
    """
    MixKV: mixing importance and diversity for per-head KV cache compression in MLLMs.

    MixKV observes that attention-based importance scores alone over-select redundant
    visual KV pairs, and mixes three complementary scores per head:
    (1) window attention received from recent queries (importance),
    (2) negative cosine similarity of each key to the mean key (diversity, KeyDiff-style),
    (3) the L2 norm of the value states (importance).
    The diversity and value-norm scores are min-max normalized and rescaled to the
    attention score's mean so the three terms are comparable, then averaged.

    Based on MixKV (https://arxiv.org/abs/2511.03878),
    official implementation: https://github.com/xuyang-liu16/MixKV.

    Adaptation notes: this implements the flagship `mixkv` scoring of the official repo
    with uniform per-head budgets (the head-wise budget variant relies on the same
    offline visual-head assets as SparseMM, see methods.sparsemm). The most recent
    `window_size` tokens are always kept.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    window_size : int, default=32
        Number of recent queries used for the attention score; these positions are
        always kept.
    kernel_size : int, default=5
        Pooling kernel applied to the attention scores.
    pooling : str, default="avgpool"
        Pooling method, "avgpool" or "maxpool".
    """

    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"

    def score(self, ctx: LayerContext) -> torch.Tensor:
        window = min(self.window_size, ctx.seq_len - 1)

        # Importance: mean attention received from the observation window (past tokens only)
        attn_scores = ctx.to_kv_heads(ctx.window_attention(window)[..., :-window].mean(dim=-2))
        attn_scores = smooth_scores(attn_scores, self.kernel_size, self.pooling)
        attn_mean = attn_scores.mean(dim=-1, keepdim=True)

        # Diversity: keys dissimilar from the mean key are valuable (KeyDiff)
        key_norm = nn.functional.normalize(ctx.keys, dim=-1)
        key_mean = key_norm.sum(dim=2, keepdim=True) / ctx.seq_len
        similarity = torch.matmul(key_norm[:, :, :-window], key_mean.transpose(-2, -1)).squeeze(-1)
        diversity_scores = _minmax_rescale(-similarity, attn_mean)

        # Importance: value norm
        vnorm_scores = _minmax_rescale(ctx.values[:, :, :-window].norm(p=2, dim=-1), attn_mean)

        combined = (attn_scores + diversity_scores + vnorm_scores) / 3

        # The observation window is always kept
        window_scores = torch.full(
            (*combined.shape[:-1], window), torch.finfo(combined.dtype).max,
            dtype=combined.dtype, device=combined.device,
        )
        return torch.cat([combined, window_scores], dim=-1)
