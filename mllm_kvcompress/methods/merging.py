# SPDX-License-Identifier: Apache-2.0
"""Text-prior KV selection with merging of evicted visual KV pairs (LOOK-M / MEDA)."""

from typing import Optional

import torch
from torch.nn import functional as F

TEXT_PRIOR_PENALTY = 1e4


def text_prior_select_and_merge(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    vision_mask: Optional[torch.Tensor],
    n_kept: int,
    recent_fraction: float,
    merge: str,
    merge_keys: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Text-prior KV selection with optional merging of the evicted KV pairs, as introduced
    by LOOK-M (https://arxiv.org/abs/2406.18139) and reused by MEDA.

    Selection: the most recent tokens are always preserved (recent window), then the
    remaining budget goes to the KV pairs with the highest importance scores, where
    vision positions are penalized so that text KV pairs are evicted last (text-prior,
    the "anti-image mask" of the official implementation).

    Merging: instead of being discarded, each evicted KV pair can be merged into its
    most similar kept KV pair (cosine similarity of keys):
    - "pivot":    merged value is the average of the evicted and the pivot KV
    - "average":  evicted KVs are averaged into the pivot KV
    - "weighted": evicted and pivot KVs are combined weighted by their similarity
    - "none":     evicted KVs are discarded

    Parameters
    ----------
    keys, values : torch.Tensor
        Shape (batch_size, num_kv_heads, seq_len, head_dim).
    scores : torch.Tensor
        Importance scores, shape (batch_size, num_kv_heads, seq_len).
    vision_mask : torch.Tensor or None
        Boolean mask of vision positions, shape (batch_size, seq_len). If None, the
        text-prior penalty is skipped and selection is purely score-based.
    n_kept : int
        Number of KV pairs to keep.
    recent_fraction : float
        Fraction of the kept budget reserved for the most recent tokens.
    merge : str
        One of "pivot", "average", "weighted", "none".
    merge_keys : bool, default=False
        Whether to also merge the keys of evicted KV pairs (the official LOOK-M merges
        both keys and values). Disabled by default: under the multimodal rotary
        embeddings (mrope) of Qwen2-VL, keys carry 3D positional phases, and averaging
        post-RoPE keys of spatially distant tokens produces invalid keys that corrupt
        generation (measured: 6/20 vs 17/20 A-OKVQA accuracy at 50% compression).
        Key merging was validated by LOOK-M on LLaVA's standard 1D RoPE only.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Compressed keys and values of sequence length n_kept (in position order).
    """
    assert merge in ("pivot", "average", "weighted", "none"), f"Unknown merge strategy: {merge}"
    bsz, num_kv_heads, k_len, head_dim = keys.shape

    n_kept = max(1, min(n_kept, k_len))
    if n_kept == k_len:
        return keys, values

    n_recent = max(1, min(int(round(n_kept * recent_fraction)), n_kept))
    n_hh = n_kept - n_recent

    scores = scores.float()
    if vision_mask is not None:
        # Text-prior: penalize vision positions so text KV pairs are kept first
        scores = scores - vision_mask[:, None, :].float() * TEXT_PRIOR_PENALTY

    # Select heavy hitters outside the recent window, then add the recent window
    kept_idx = torch.arange(k_len - n_recent, k_len, device=keys.device)
    kept_idx = kept_idx[None, None, :].expand(bsz, num_kv_heads, n_recent)
    if n_hh > 0:
        hh_idx = scores[..., : k_len - n_recent].topk(n_hh, dim=-1).indices
        kept_idx = torch.cat([hh_idx, kept_idx], dim=-1)
    kept_idx = kept_idx.sort(dim=-1).values

    kept_idx_d = kept_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kept_keys = keys.gather(2, kept_idx_d)
    kept_values = values.gather(2, kept_idx_d)

    if merge == "none":
        return kept_keys.contiguous(), kept_values.contiguous()

    # Evicted positions, in position order (stable argsort of the keep mask)
    keep_mask = torch.zeros(bsz, num_kv_heads, k_len, dtype=torch.bool, device=keys.device)
    keep_mask.scatter_(2, kept_idx, True)
    n_pruned = k_len - n_kept
    pruned_idx = (~keep_mask).float().argsort(dim=-1, descending=True, stable=True)[..., :n_pruned]
    pruned_idx = pruned_idx.sort(dim=-1).values
    pruned_idx_d = pruned_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    pruned_keys = keys.gather(2, pruned_idx_d)
    pruned_values = values.gather(2, pruned_idx_d)

    # Assign each evicted KV pair to its most similar kept key (cosine similarity).
    # Targets are restricted to kept *vision* positions: merging into text or recent
    # KV pairs corrupts the prompt structure and degrades generation badly. (The
    # official LOOK-M code merges into any kept position, but the paper describes
    # merging evicted visual KVs into retained visual KVs.)
    similarity = torch.matmul(F.normalize(pruned_keys, dim=-1), F.normalize(kept_keys, dim=-1).transpose(-1, -2))
    if vision_mask is not None:
        kept_is_vision = vision_mask[:, None, :].expand(-1, num_kv_heads, -1).gather(2, kept_idx)
        if not kept_is_vision.any():
            return kept_keys.contiguous(), kept_values.contiguous()
        similarity = similarity.masked_fill(~kept_is_vision[:, :, None, :], float("-inf"))
    max_sim, pivot_idx = similarity.max(dim=-1)
    pivot_idx_d = pivot_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)

    pivot_keys = kept_keys.gather(2, pivot_idx_d)
    pivot_values = kept_values.gather(2, pivot_idx_d)

    if merge == "pivot":
        src_keys = (pruned_keys + pivot_keys) / 2
        src_values = (pruned_values + pivot_values) / 2
    elif merge == "average":
        src_keys = pruned_keys
        src_values = pruned_values
    else:  # weighted
        weight = max_sim.unsqueeze(-1).to(keys.dtype)
        src_keys = weight * pruned_keys + (1 - weight) * pivot_keys
        src_values = weight * pruned_values + (1 - weight) * pivot_values

    if merge_keys:
        kept_keys = kept_keys.scatter_reduce(2, pivot_idx_d, src_keys, reduce="mean", include_self=True)
    kept_values = kept_values.scatter_reduce(2, pivot_idx_d, src_values, reduce="mean", include_self=True)

    return kept_keys.contiguous(), kept_values.contiguous()
