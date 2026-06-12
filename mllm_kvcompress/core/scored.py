# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.getLogger(__name__)


@dataclass
class ScoredEviction(CompressionMethod):
    """
    Base class for score-then-evict compression methods.

    Subclasses implement `score(ctx)` returning per-KV-pair importance scores of shape
    (batch_size, num_kv_heads, seq_len); the lowest-scored fraction `ratio` of KV
    pairs is evicted per head. Modality-aware methods can use `ctx.vision_mask` and
    `ctx.vision_token_spans` to score text and vision positions differently.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to evict during pre-filling.
    """

    ratio: float = 0.0

    def __post_init__(self):
        assert 0 <= self.ratio < 1, "ratio must be in [0, 1)"

    def score(self, ctx: LayerContext) -> torch.Tensor:
        raise NotImplementedError

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ratio == 0:
            return ctx.keys, ctx.values

        scores = self.score(ctx)
        n_kept = int(ctx.seq_len * (1 - self.ratio))
        indices = scores.topk(n_kept, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, ctx.head_dim)

        keys = ctx.keys.gather(2, indices).contiguous()
        values = ctx.values.gather(2, indices).contiguous()
        return keys, values
