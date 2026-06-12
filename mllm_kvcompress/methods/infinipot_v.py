# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from torch import nn
from transformers.utils import logging

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


@dataclass
class InfiniPotV(CompressionMethod):
    """
    InfiniPot-V: memory-constrained KV cache compression for streaming video.

    InfiniPot-V compresses the *vision* KV pairs with two complementary criteria:
    (1) TaR (Temporal-axis Redundancy): vision keys that are similar to the keys of the
        most recent video segment (the "query" segment) are redundant; tokens with low
        cosine similarity to the recent segment are distinctive and kept.
    (2) VaN (Value Norm): among the remaining tokens, those with the largest value
        norms are kept.
    Text (system/instruction) KV pairs are never compressed.

    Based on InfiniPot-V (https://arxiv.org/abs/2506.15745, NeurIPS 2025),
    official implementation: https://github.com/aiha-lab/InfiniPot-V.

    Adaptation notes: the official implementation processes streaming video in blocks,
    re-compressing the cache whenever a memory cap is hit, with per-frame token counts.
    This library compresses one pre-fill, so this is the single-shot core of the
    method: the "query segment" is the most recent `query_ratio` fraction of vision
    tokens (instead of the most recent frames), and similarity is computed
    token-to-segment rather than frame-to-frame. The streaming loop and the
    layer-adaptive 1D/2D/3D pooling of VaN scores (which requires the video grid
    shape) are not included.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of *vision* key-value pairs to remove. Text KV pairs are never removed.
    tar_ratio : float, default=0.5
        Fraction of the kept-vision budget allocated by TaR (the rest is VaN).
    query_ratio : float, default=0.25
        Fraction of the most recent vision tokens used as the TaR query segment
        (always kept as part of the TaR budget).
    """

    ratio: float = 0.0
    tar_ratio: float = 0.5
    query_ratio: float = 0.25

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"
        assert 0 <= self.tar_ratio <= 1 and 0 < self.query_ratio < 1

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values
        if ctx.vision_mask is None:
            logger.warning_once("InfiniPotV could not find vision token positions, no compression is applied.")
            return ctx.keys, ctx.values

        keys, values = ctx.keys, ctx.values
        keys_out, values_out = [], []
        for batch_idx in range(ctx.batch_size):
            vision_positions = ctx.vision_mask[batch_idx].nonzero().squeeze(-1)
            text_positions = (~ctx.vision_mask[batch_idx]).nonzero().squeeze(-1)
            n_vision = vision_positions.numel()
            n_keep = int(n_vision * (1 - self.ratio))
            if n_vision == 0 or n_keep >= n_vision:
                keys_out.append(keys[batch_idx : batch_idx + 1])
                values_out.append(values[batch_idx : batch_idx + 1])
                continue

            vision_keys = keys[batch_idx, :, vision_positions]  # (h, n_vision, d)
            vision_values = values[batch_idx, :, vision_positions]

            query_len = max(1, int(n_vision * self.query_ratio))
            tar_budget = round(self.tar_ratio * n_keep)

            # TaR: keep the query segment plus the earlier tokens least similar to it
            tar_indices = None
            if tar_budget > query_len:
                key_norm = nn.functional.normalize(vision_keys, dim=-1)
                query_mean = key_norm[:, -query_len:].mean(dim=1, keepdim=True)  # (h, 1, d)
                redundancy = -(key_norm[:, :-query_len] * query_mean).sum(dim=-1)
                distinct = redundancy.topk(tar_budget - query_len, dim=-1).indices
                recent = (
                    torch.arange(n_vision - query_len, n_vision, device=keys.device)
                    .expand(ctx.num_kv_heads, -1)
                )
                tar_indices = torch.cat([distinct, recent], dim=-1)

            # VaN: value-norm score, with TaR selections forced to the top
            van_scores = vision_values.norm(p=2, dim=-1)  # (h, n_vision)
            if tar_indices is not None:
                van_scores = van_scores.scatter(
                    -1, tar_indices, torch.full_like(tar_indices, 1, dtype=van_scores.dtype) * (van_scores.max() + 1)
                )
            kept_vision = van_scores.topk(n_keep, dim=-1).indices.sort(dim=-1).values  # (h, n_keep)

            # Final per-head index set: all text positions + kept vision positions, in order
            kept_global = vision_positions[kept_vision]  # (h, n_keep)
            text_expand = text_positions.expand(ctx.num_kv_heads, -1)
            kept = torch.cat([text_expand, kept_global], dim=-1).sort(dim=-1).values

            idx = kept[None, :, :, None].expand(1, ctx.num_kv_heads, -1, ctx.head_dim)
            keys_out.append(keys[batch_idx : batch_idx + 1].gather(2, idx))
            values_out.append(values[batch_idx : batch_idx + 1].gather(2, idx))

        return torch.cat(keys_out, dim=0).contiguous(), torch.cat(values_out, dim=0).contiguous()
