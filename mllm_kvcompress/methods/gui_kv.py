# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.attention import smooth_scores
from mllm_kvcompress.core.scored import ScoredEviction
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


@dataclass
class GUIKV(ScoredEviction):
    """
    GUI-KV: KV cache compression with spatial saliency and temporal redundancy for GUI agents.

    GUI-KV augments SnapKV-style attention scores with two GUI-specific signals:
    (1) Spatial saliency: the L2 norm of the hidden states of the current screenshot's
        vision tokens identifies semantically important GUI elements; a temperature-
        softmaxed saliency is added to the attention scores with weight `alpha`.
    (2) Temporal redundancy: with multiple screenshots in context, the keys of the
        current screenshot span a subspace (via QR decomposition); previous-screenshot
        keys that are well explained by this subspace (small projection residual) are
        redundant and their scores are zeroed.

    Based on GUI-KV (https://arxiv.org/abs/2510.00536),
    official implementation: https://github.com/SalesforceAIResearch/GUI-KV.

    Adaptation notes: scores are computed at KV-head granularity (GQA mean) and the QR
    redundancy test uses the KV-head keys directly instead of group-repeated heads.

    Parameters
    ----------
    ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    alpha : float, default=0.1
        Weight of the spatial saliency score.
    temperature : float, default=1.0
        Softmax temperature for the spatial saliency normalization.
    qr_rank : int, default=32
        Rank of the current-screenshot key subspace used for redundancy detection.
    window_size : int, default=32
        Number of recent queries used for the attention scores; always kept.
    kernel_size : int, default=5
        Pooling kernel applied to the attention scores.
    pooling : str, default="avgpool"
        Pooling method, "avgpool" or "maxpool".
    """

    alpha: float = 0.1
    temperature: float = 1.0
    qr_rank: int = 32
    window_size: int = 32
    kernel_size: int = 5
    pooling: str = "avgpool"

    def score(self, ctx: LayerContext) -> torch.Tensor:
        window = min(self.window_size, ctx.seq_len - 1)
        scores = ctx.to_kv_heads(ctx.window_attention(window).sum(dim=-2))
        scores = smooth_scores(scores, self.kernel_size, self.pooling)

        if ctx.vision_mask is None:
            logger.warning_once(
                "GUIKV could not find vision token positions, falling back to "
                "SnapKV-style attention-only scoring."
            )

        for batch_idx, batch_spans in enumerate(ctx.vision_token_spans):
            if not batch_spans:
                continue
            start, end = batch_spans[-1]

            # Spatial saliency: standardized, temperature-softmaxed hidden state norms
            # of the current (last) screenshot, added with weight alpha
            saliency = ctx.hidden_states[batch_idx, start:end].norm(p=2, dim=-1).float()
            saliency = (saliency - saliency.mean()) / (saliency.std() + 1e-8)
            saliency = torch.softmax(saliency / self.temperature, dim=0).to(scores.dtype)
            scores[batch_idx, :, start:end] += self.alpha * saliency

            # Temporal redundancy: zero the scores of previous-screenshot keys that lie
            # in the subspace spanned by the current screenshot's keys
            if len(batch_spans) > 1:
                budget = 1 - self.ratio
                for head_idx in range(ctx.num_kv_heads):
                    current = ctx.keys[batch_idx, head_idx, start:end].float()
                    rank = min(self.qr_rank, *current.shape)
                    Q = torch.linalg.qr(current.T, mode="reduced")[0][:, :rank]

                    residual_norms, positions = [], []
                    for prev_start, prev_end in batch_spans[:-1]:
                        prev = ctx.keys[batch_idx, head_idx, prev_start:prev_end].float()
                        residual = prev - (Q @ (Q.T @ prev.T)).T
                        residual_norms.append(residual.norm(p=2, dim=-1))
                        positions.append(torch.arange(prev_start, prev_end, device=ctx.keys.device))
                    residual_norms = torch.cat(residual_norms)
                    positions = torch.cat(positions)

                    n_redundant = int((1 - budget) * len(residual_norms))
                    if n_redundant > 0:
                        redundant = residual_norms.topk(n_redundant, largest=False).indices
                        scores[batch_idx, head_idx, positions[redundant]] = 0

        # The observation window is always kept
        scores[..., -window:] = torch.finfo(scores.dtype).max
        return scores
