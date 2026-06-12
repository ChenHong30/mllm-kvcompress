# SPDX-License-Identifier: Apache-2.0
"""
Attention utilities for multimodal language models: multimodal rotary embeddings
(mrope), observation-window attention scores and head-wise key masking.
"""

import math

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
    (batch, num_kv_heads, seq_len, head_dim) -> (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def apply_rope_to_queries(
    query_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int] | None,
) -> torch.Tensor:
    """
    Apply rotary position embeddings to query states, supporting both the multimodal
    rotary embeddings (mrope) used by Qwen2-VL / Qwen2.5-VL and standard RoPE.

    For mrope, cos/sin have shape (3, batch_size, seq_len, head_dim) where the leading
    dimension indexes the temporal / height / width axes, and `mrope_section` gives the
    channel split (in pairs) across the three axes.
    """
    if cos.dim() == 4:
        assert mrope_section is not None, "mrope_section is required for multimodal rotary embeddings"
        section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(section, dim=-1))], dim=-1).unsqueeze(1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(section, dim=-1))], dim=-1).unsqueeze(1)
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rot, query_pass = query_states[..., :rotary_dim], query_states[..., rotary_dim:]
    query_rot = (query_rot * cos) + (rotate_half(query_rot) * sin)
    if query_pass.numel() == 0:
        return query_rot
    return torch.cat([query_rot, query_pass], dim=-1)


def get_query_states(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Extract (pre-RoPE) query states (batch, num_heads, seq_len, head_dim) from an attention module."""
    if not hasattr(module, "q_proj"):
        raise NotImplementedError(f"Query extraction not implemented for {module.__class__.__name__}")
    bsz, q_len, _ = hidden_states.shape
    num_heads = module.config.num_attention_heads
    query_size = num_heads * module.head_dim
    query_states = module.q_proj(hidden_states)
    if query_states.shape[-1] == query_size * 2:
        # Qwen3.5 packs each head as [query, gate].
        query_states = query_states.view(bsz, q_len, num_heads, module.head_dim * 2)[..., : module.head_dim]
    else:
        query_states = query_states[..., :query_size].view(bsz, q_len, num_heads, module.head_dim)
    if hasattr(module, "q_norm"):
        query_states = module.q_norm(query_states)
    return query_states.transpose(1, 2)


def get_mrope_section(module: nn.Module) -> list[int] | None:
    """Return the mrope channel split of an attention module, or None for standard RoPE models."""
    # rope_scaling was renamed to rope_parameters in transformers v5
    rope_params = (
        getattr(module, "rope_parameters", None)
        or getattr(module.config, "rope_parameters", None)
        or getattr(module, "rope_scaling", None)
        or getattr(module.config, "rope_scaling", None)
    )
    if rope_params is None:
        return None
    return rope_params.get("mrope_section")


def observation_window_attention(
    module: nn.Module,
    hidden_states: torch.Tensor,
    keys: torch.Tensor,
    window_size: int,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """
    Recompute the attention weights of the last `window_size` queries over all keys.

    Most official MLLM compression implementations rely on the full eager pre-fill
    attention map, which is unavailable with sdpa/flash attention. Importance scores
    are therefore approximated from an observation window of recent queries (SnapKV
    style), which only costs O(window_size * seq_len).

    Returns causal attention weights of shape (batch_size, num_heads, window_size, seq_len).
    """
    bsz, _, k_len, _ = keys.shape
    window_size = min(window_size, hidden_states.shape[1])
    query_states = get_query_states(module, hidden_states[:, -window_size:])

    cos, sin = position_embeddings
    cos, sin = cos[..., -window_size:, :], sin[..., -window_size:, :]
    query_states = apply_rope_to_queries(query_states, cos, sin, get_mrope_section(module))

    num_key_value_groups = module.config.num_attention_heads // module.config.num_key_value_heads
    key_states = repeat_kv(keys, num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(module.head_dim)
    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
    attn_weights += attention_mask
    return nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)


def group_mean(scores: torch.Tensor, num_key_value_heads: int) -> torch.Tensor:
    """
    Average scores over grouped query heads (GQA).
    (batch_size, num_heads, seq_len) -> (batch_size, num_kv_heads, seq_len)
    """
    bsz, num_heads, k_len = scores.shape
    return scores.view(bsz, num_key_value_heads, num_heads // num_key_value_heads, k_len).mean(2)


def smooth_scores(scores: torch.Tensor, kernel_size: int, pooling: str) -> torch.Tensor:
    """Smooth importance scores across neighboring tokens (SnapKV-style pooling)."""
    if kernel_size <= 1 or scores.shape[-1] < kernel_size:
        return scores
    if pooling == "avgpool":
        return nn.functional.avg_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    if pooling == "maxpool":
        return nn.functional.max_pool1d(scores, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
    raise ValueError(f"Pooling method not supported: {pooling}")


# ---------------------------------------------------------------------------
# Head-wise key masking ("virtual eviction" for methods with per-head budgets)
# ---------------------------------------------------------------------------


def search_hyperplane(X: torch.Tensor, max_iter: int = 1000) -> torch.Tensor:
    """
    Given a tensor X of shape (bsz, seq_len, head_dim), search for a hyperplane Y
    (bsz, head_dim) such that for every i, <X[:, i], Y> <= 0. Returns -1e5 * Y / ||Y||²
    to ensure exp(<X, Y>) = 0. Raises a ValueError if no such hyperplane is found.
    """
    Y = X.mean(1)  # this initialization is enough for most cases
    for _ in range(max_iter):
        mask = torch.bmm(X, Y.unsqueeze(-1)) <= 0
        if not mask.any():
            return -1e5 * Y / Y.norm(dim=-1, keepdim=True) ** 2
        Y += (X * mask).sum(1) / mask.sum(1).clamp(min=1)
    raise ValueError("Could not find fake keys such that for every query q, exp(<q, k>) = 0")


def headwise_masking(func):
    """
    Wrap an attention function to neutralize the keys listed in
    module.evicted_key_indices before the attention computation. The keys are replaced
    by a fake key k such that exp(<q, k>) = 0, emulating head-wise eviction for methods
    with per-head budgets (e.g. SparseMM). This does not reduce peak memory but
    reproduces the attention sparsity exactly.
    """

    def wrapper(module, query, key, value, attention_mask, dropout=0.0, **kwargs):
        if query.shape[2] == key.shape[2]:
            # Pre-filling
            module.evicted_key_indices = None
        elif getattr(module, "evicted_key_indices", None) is not None:
            # Decoding: build fake keys k s.t. exp(<q, k>) = 0
            bsz, num_heads, seq_len, head_dim = query.shape
            num_key_value_heads = key.shape[1]
            num_groups = num_heads // num_key_value_heads

            q = query.view(bsz, num_key_value_heads, num_groups, seq_len, head_dim)
            q = q.reshape(bsz * num_key_value_heads, num_groups * seq_len, head_dim)
            k = search_hyperplane(q.float()).to(key.dtype)
            k = k.view(bsz, num_key_value_heads, head_dim)

            batch_indices, head_indices, seq_indices = module.evicted_key_indices
            key[batch_indices, head_indices, seq_indices] = k[batch_indices, head_indices]

        if "cu_seq_lens_k" in kwargs and kwargs["cu_seq_lens_k"] is not None:
            kwargs["cu_seq_lens_k"][-1] = key.shape[-2]
        return func(module, query, key, value, attention_mask, dropout, **kwargs)

    wrapper._mllm_kvcompress_headwise = True
    return wrapper


def enable_headwise_masking():
    """Patch all registered attention functions to support head-wise key masking. Idempotent."""
    for name, func in ALL_ATTENTION_FUNCTIONS.items():
        if not getattr(func, "_mllm_kvcompress_headwise", False):
            ALL_ATTENTION_FUNCTIONS[name] = headwise_masking(func)
