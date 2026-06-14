# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
from transformers.utils import logging

from mllm_kvcompress.core.attention import (
    apply_rope_to_queries,
    get_mrope_section,
    get_query_states,
    repeat_kv,
)
from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


@dataclass
class FitPrune(CompressionMethod):
    """
    FitPrune: progressive visual token pruning with a per-layer removal schedule.

    FitPrune removes visual tokens layer by layer. At each layer, a token's importance
    is the product of (1) the self-attention it receives from the other visual tokens
    and (2) the cross-attention it receives from the subsequent text tokens; the
    lowest-scored tokens are removed according to a per-layer deletion schedule, and a
    token removed at layer l stays removed at all deeper layers.

    Based on FitPrune (https://arxiv.org/abs/2409.10197, AAAI 2025),
    official implementation: https://github.com/ywh187/FitPrune.

    Adaptation notes:
    - The official deletion schedule is *fitted offline* on a calibration set (binary
      search minimizing the attention distribution divergence) and hard-coded per model
      and ratio. The fitting procedure is not included here: by default deletions are
      spread uniformly over the layers after `start_layer`; a fitted schedule can be
      passed via `delete_schedule`.
    - The official implementation prunes hidden states (removing tokens from all
      subsequent compute). As a cache compression method, we drop the corresponding
      KV pairs at each layer instead.

    Parameters
    ----------
    ratio : float, default=0.0
        Total fraction of *vision* tokens removed by the end of the network.
    delete_schedule : list[int], optional
        Number of vision tokens to delete at each layer (length = num layers). When
        provided, overrides ratio / start_layer.
    start_layer : int, default=2
        First layer at which tokens are deleted (official fitted schemes delete
        almost nothing before layer 2).
    """

    ratio: float = 0.0
    delete_schedule: Optional[list[int]] = None
    start_layer: int = 2

    _kept_vision: Optional[torch.Tensor] = field(default=None, init=False, repr=False)
    _schedule: Optional[list[int]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"

    def _init_schedule(self, num_layers: int, n_vision: int, compressible_layers: list[int]):
        if self.delete_schedule is not None:
            self._schedule = list(self.delete_schedule)
            return
        total = int(n_vision * self.ratio)
        # Spread deletions over the layers that actually compress (full-attention layers
        # for hybrid models such as Qwen3.5, every layer for dense models) from
        # start_layer onward, so the target ratio is reached regardless of the layout.
        active = [i for i in compressible_layers if i >= self.start_layer] or compressible_layers or [num_layers - 1]
        per_layer = total // len(active)
        schedule = [0] * num_layers
        for layer_idx in active:
            schedule[layer_idx] = per_layer
        schedule[active[-1]] += total - per_layer * len(active)
        self._schedule = schedule

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values

        if self._schedule is None:
            # Lazy initialization on the first compressed layer. Keying off the schedule
            # being unset (rather than layer_idx == 0) keeps this robust to models whose
            # first full-attention layer is not index 0, e.g. the hybrid Qwen3.5 whose
            # full-attention layers are [3, 7, 11, ...].
            if ctx.batch_size > 1:
                logger.warning_once("FitPrune currently supports batch_size=1, no compression applied.")
                return ctx.keys, ctx.values
            if ctx.vision_mask is None:
                logger.warning_once("FitPrune could not find vision token positions, no compression is applied.")
                return ctx.keys, ctx.values
            layer_types = getattr(ctx.module.config, "layer_types", None)
            if layer_types is not None:
                compressible = [i for i, kind in enumerate(layer_types) if "full" in str(kind).lower()]
            else:
                compressible = list(range(ctx.num_layers))
            self._kept_vision = ctx.vision_mask[0].nonzero().squeeze(-1)
            self._init_schedule(ctx.num_layers, self._kept_vision.numel(), compressible)

        n_delete = min(self._schedule[ctx.layer_idx], max(0, self._kept_vision.numel() - 1))
        if n_delete > 0:
            vis_pos = self._kept_vision
            num_groups = ctx.module.config.num_attention_heads // ctx.num_kv_heads
            keys_full = repeat_kv(ctx.keys, num_groups)

            # Self-attention: visual queries over all earlier positions (causal), then
            # restricted to the kept visual columns
            queries = get_query_states(ctx.module, ctx.hidden_states[:, vis_pos])
            cos, sin = ctx.position_embeddings
            queries = apply_rope_to_queries(
                queries, cos[..., vis_pos, :], sin[..., vis_pos, :], get_mrope_section(ctx.module)
            )
            attn = torch.matmul(queries, keys_full.transpose(2, 3)) / math.sqrt(ctx.head_dim)
            causal = torch.arange(ctx.seq_len, device=ctx.keys.device)[None, :] > vis_pos[:, None]
            attn = attn.masked_fill(causal[None, None], float("-inf"))
            attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
            attn = attn.max(dim=1).values[0]  # head-max, (n_vis, seq_len)
            self_score = attn[:, vis_pos].sum(dim=0) / vis_pos.numel()

            # Cross-attention: every text query after the visual block over the kept
            # visual columns (head-max, summed over the text rows), as in the reference.
            last_vision = int(ctx.vision_mask[0].nonzero().max())
            text_pos = torch.arange(last_vision + 1, ctx.seq_len, device=ctx.keys.device)
            if text_pos.numel() > 0:
                text_q = get_query_states(ctx.module, ctx.hidden_states[:, text_pos])
                text_q = apply_rope_to_queries(
                    text_q, cos[..., text_pos, :], sin[..., text_pos, :], get_mrope_section(ctx.module)
                )
                cross_attn = torch.matmul(text_q, keys_full.transpose(2, 3)) / math.sqrt(ctx.head_dim)
                cross_causal = torch.arange(ctx.seq_len, device=ctx.keys.device)[None, :] > text_pos[:, None]
                cross_attn = cross_attn.masked_fill(cross_causal[None, None], float("-inf"))
                cross_attn = nn.functional.softmax(cross_attn, dim=-1, dtype=torch.float32)
                cross_score = cross_attn.max(dim=1).values[0][:, vis_pos].sum(dim=0)
            else:
                cross_score = torch.ones_like(self_score)

            scores = self_score * cross_score.float()
            dropped = scores.topk(n_delete, largest=False).indices
            keep_mask = torch.ones_like(vis_pos, dtype=torch.bool)
            keep_mask[dropped] = False
            self._kept_vision = vis_pos[keep_mask]

        # Gather the kept positions (all text + surviving vision tokens) for this layer
        if self._kept_vision.numel() == int(ctx.vision_mask[0].sum()):
            return ctx.keys, ctx.values
        kept = torch.cat([(~ctx.vision_mask[0]).nonzero().squeeze(-1), self._kept_vision]).sort().values
        idx = kept[None, None, :, None].expand(ctx.batch_size, ctx.num_kv_heads, -1, ctx.head_dim)
        return ctx.keys.gather(2, idx).contiguous(), ctx.values.gather(2, idx).contiguous()
