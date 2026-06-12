# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.attention import smooth_scores
from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


def spatial_mutual_information(
    probs: torch.Tensor, n_row_bins: int = 12, n_col_bins: int = 12, n_val_bins: int = 16
) -> float:
    """
    Mutual information between a (normalized, non-negative) attention distribution over
    vision tokens and their 2D spatial position on an inferred square-ish grid.
    High MI means the head's attention is spatially structured (informative head).
    """
    n = probs.numel()
    if n <= 1 or float(probs.sum()) <= 1e-12:
        return 0.0
    grid_h = max(1, round(math.sqrt(n)))
    grid_w = max(1, (n + grid_h - 1) // grid_h)
    pos = torch.arange(n, device=probs.device)
    row_bins = (((pos // grid_w).float() + 0.5) / grid_h * n_row_bins).long().clamp(0, n_row_bins - 1)
    col_bins = (((pos % grid_w).float() + 0.5) / grid_w * n_col_bins).long().clamp(0, n_col_bins - 1)
    y_bins = row_bins * n_col_bins + col_bins
    if y_bins.unique().numel() <= 1:
        return 0.0

    # Value bins from the rank of each token's probability
    rank = probs.argsort().argsort()
    a_bins = (rank * n_val_bins // n).clamp(0, n_val_bins - 1)
    if a_bins.unique().numel() <= 1:
        return 0.0

    joint = torch.zeros(n_val_bins, n_row_bins * n_col_bins, device=probs.device, dtype=torch.float64)
    joint.index_put_((a_bins, y_bins), torch.ones(n, device=probs.device, dtype=torch.float64), accumulate=True)
    joint = joint / joint.sum()
    pa = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    mi = (joint * (torch.log(joint + 1e-12) - torch.log(pa @ py + 1e-12))).sum()
    return max(0.0, float(mi))


@dataclass
class STaRKV(CompressionMethod):
    """
    STaR-KV: spatially/structurally-aware re-weighted KV cache compression for MLLMs.

    STaR-KV builds on SnapKV-style observation-window scores shared across heads, and
    re-weights them with three signals:
    (1) Spatial saliency: hidden-state norms of the vision tokens (as in GUI-KV).
    (2) MI group prior: per KV-head-group attention distributions over the vision grid
        are scored by their spatial mutual information; tokens dominated by spatially
        informative groups are up-weighted (softmax(MI / mi_tau), blended with weight
        `mi_lambda`).
    (3) AEB sharpening: the score distribution is sharpened (score ** 1/T) with a
        temperature derived from its normalized entropy.

    Based on STaR-KV, official implementation: https://github.com/kawhiiiileo/STaR-KV.

    Adaptation notes: the official implementation maintains EMA statistics (MI prior,
    temporal pattern, AEB temperature) across the steps of a GUI-agent episode. This
    library compresses one pre-fill at a time, so the MI prior and AEB temperature are
    computed from the current pre-fill only, and the cross-step temporal discount
    (a no-op for single-image prompts in the official code) is not applied.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    alpha : float, default=0.1
        Weight of the spatial saliency score.
    temperature : float, default=1.0
        Softmax temperature for the spatial saliency normalization.
    mi_lambda : float, default=0.5
        Blend weight of the MI group prior (0 disables it).
    mi_tau : float, default=1.0
        Softmax temperature mapping group MI to group weights.
    aeb_min_temp, aeb_max_temp : float, defaults=0.95 / 1.05
        Temperature range for entropy-guided score sharpening.
    window_size : int, default=32
        Number of recent queries used for the attention scores; always kept.
    kernel_size : int, default=5
        Pooling kernel applied to the attention scores.
    pooling : str, default="avgpool"
        Pooling method, "avgpool" or "maxpool".
    """

    ratio: float = 0.0
    alpha: float = 0.1
    temperature: float = 1.0
    mi_lambda: float = 0.5
    mi_tau: float = 1.0
    aeb_min_temp: float = 0.95
    aeb_max_temp: float = 1.05
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert 0 <= self.mi_lambda <= 1, "mi_lambda must be between 0 and 1"

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values

        window = min(self.window_size, ctx.seq_len - 1)
        num_groups = ctx.module.config.num_attention_heads // ctx.module.config.num_key_value_heads

        # pooled scores at query-head granularity, past tokens only
        attn_cache = smooth_scores(
            ctx.window_attention(window).sum(dim=-2), self.kernel_size, self.pooling
        )[..., :-window]

        if ctx.vision_mask is None:
            logger.warning_once(
                "STaRKV could not find vision token positions, falling back to "
                "SnapKV-style attention-only scoring."
            )
        spans = ctx.vision_token_spans

        n_kept_past = max(1, int(ctx.seq_len * (1 - self.ratio)) - window)
        all_indices = []
        for batch_idx in range(ctx.batch_size):
            ac = attn_cache[batch_idx]  # (num_heads, past_len)

            # Spatial saliency boost on the last vision span
            batch_spans = [(s, e) for s, e in spans[batch_idx] if e <= ac.shape[-1]]
            if batch_spans and self.alpha > 0:
                start, end = batch_spans[-1]
                saliency = ctx.hidden_states[batch_idx, start:end].norm(p=2, dim=-1).float()
                saliency = (saliency - saliency.mean()) / (saliency.std() + 1e-8)
                saliency = torch.softmax(saliency / self.temperature, dim=0).to(ac.dtype)
                ac = ac.clone()
                ac[:, start:end] += self.alpha * saliency

            base = ac.mean(dim=0)  # (past_len,)
            group_scores = ac.view(ctx.num_kv_heads, num_groups, -1).mean(dim=1)

            # MI group prior over the last vision span
            scaled = base
            if batch_spans and self.mi_lambda > 0:
                start, end = batch_spans[-1]
                vis = group_scores[:, start:end].float().clamp_min(0)
                mi = torch.tensor(
                    [spatial_mutual_information(g / g.sum().clamp_min(1e-12)) for g in vis],
                    device=base.device,
                )
                if float(mi.max() - mi.min()) < 1e-12:
                    normalized_mi = torch.full_like(mi, 0.5)
                else:
                    normalized_mi = (mi - mi.min()) / (mi.max() - mi.min())
                group_weights = torch.softmax(normalized_mi / max(self.mi_tau, 1e-6), dim=0).to(base.dtype)
                dominant_group = group_scores.argmax(dim=0)  # (past_len,)
                token_weight = group_weights[dominant_group] * ctx.num_kv_heads
                scaled = base * ((1 - self.mi_lambda) + self.mi_lambda * token_weight)

            # AEB: entropy-guided sharpening
            probs = scaled.float().clamp_min(0)
            probs = probs / probs.sum().clamp_min(1e-8)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            normalized_entropy = entropy / math.log(max(2, probs.numel()))
            aeb_temp = self.aeb_min_temp + (self.aeb_max_temp - self.aeb_min_temp) * float(normalized_entropy)
            scaled = scaled.clamp_min(0) ** (1 / aeb_temp)

            kept = scaled.topk(min(n_kept_past, scaled.numel())).indices.sort().values
            all_indices.append(kept)

        # Shared indices across heads; per-batch gather then re-stack (budgets are equal)
        keys_out, values_out = [], []
        for batch_idx, kept in enumerate(all_indices):
            idx = kept[None, None, :, None].expand(1, ctx.num_kv_heads, -1, ctx.head_dim)
            k_past = ctx.keys[batch_idx : batch_idx + 1, :, :-window].gather(2, idx)
            v_past = ctx.values[batch_idx : batch_idx + 1, :, :-window].gather(2, idx)
            keys_out.append(torch.cat([k_past, ctx.keys[batch_idx : batch_idx + 1, :, -window:]], dim=2))
            values_out.append(torch.cat([v_past, ctx.values[batch_idx : batch_idx + 1, :, -window:]], dim=2))
        return torch.cat(keys_out, dim=0).contiguous(), torch.cat(values_out, dim=0).contiguous()
