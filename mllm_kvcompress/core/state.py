# SPDX-License-Identifier: Apache-2.0
"""Per-layer compression context handed to compression methods."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from mllm_kvcompress.core.attention import group_mean, observation_window_attention
from mllm_kvcompress.core.modality import vision_spans


@dataclass
class LayerContext:
    """
    Everything a compression method can see when one attention layer finishes
    pre-filling. Methods receive a LayerContext and return compressed (keys, values).

    Attributes
    ----------
    module : nn.Module
        The attention layer of the language model.
    hidden_states : torch.Tensor
        Input to the attention layer, shape (batch_size, seq_len, hidden_dim).
    keys, values : torch.Tensor
        KV pairs of this layer's cache, shape (batch_size, num_kv_heads, seq_len, head_dim).
    attentions : torch.Tensor or None
        Full attention weights when the attention implementation returns them
        (eager); None for sdpa/flash attention.
    vision_mask : torch.Tensor or None
        Boolean mask (batch_size, seq_len) of vision token positions, or None if
        modality information could not be captured.
    kwargs : dict
        Raw keyword arguments of the attention forward pass.
    """

    module: nn.Module
    hidden_states: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    attentions: Optional[torch.Tensor]
    vision_mask: Optional[torch.Tensor]
    kwargs: dict

    # ---- shapes ----

    @property
    def layer_idx(self) -> int:
        return self.module.layer_idx

    @property
    def num_layers(self) -> int:
        return self.module.config.num_hidden_layers

    @property
    def batch_size(self) -> int:
        return self.keys.shape[0]

    @property
    def num_kv_heads(self) -> int:
        return self.keys.shape[1]

    @property
    def seq_len(self) -> int:
        return self.keys.shape[2]

    @property
    def head_dim(self) -> int:
        return self.keys.shape[3]

    @property
    def position_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kwargs["position_embeddings"]

    # ---- modality ----

    @property
    def vision_token_spans(self) -> list[list[tuple[int, int]]]:
        """Contiguous (start, end) vision-token runs per batch element ([] without modality info)."""
        if self.vision_mask is None:
            return [[] for _ in range(self.batch_size)]
        return vision_spans(self.vision_mask)

    # ---- attention scores ----

    def window_attention(self, window_size: int) -> torch.Tensor:
        """
        Attention weights of the last `window_size` queries over all keys, shape
        (batch_size, num_heads, window, seq_len). Uses the model's attention weights
        when available (eager), otherwise recomputes them (observation-window
        approximation, mrope-aware).
        """
        if self.attentions is not None:
            return self.attentions[:, :, -window_size:, :]
        return observation_window_attention(
            self.module, self.hidden_states, self.keys, window_size, self.position_embeddings
        )

    def to_kv_heads(self, scores: torch.Tensor) -> torch.Tensor:
        """Reduce per-query-head scores to KV-head granularity (GQA group mean)."""
        return group_mean(scores, self.module.config.num_key_value_heads)
