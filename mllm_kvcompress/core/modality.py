# SPDX-License-Identifier: Apache-2.0
"""Modality tracking: which cache positions hold vision (image/video) tokens."""

import torch


def vision_mask_from_input_ids(input_ids: torch.Tensor, config, model=None) -> torch.Tensor | None:
    """
    Boolean mask (batch_size, seq_len) of vision token positions, derived from the
    image/video placeholder token ids of the model config. Returns None when the
    config does not define vision token ids.
    """
    candidates = []
    if model is not None:
        candidates.extend(
            getattr(model, name, None)
            for name in ("img_context_token_id", "image_token_id", "video_token_id", "vision_token_id")
        )
    vision_token_ids = [
        token_id
        for token_id in (
            *candidates,
            getattr(config, "image_token_id", None),
            getattr(config, "video_token_id", None),
            getattr(config, "vision_token_id", None),
        )
        if token_id is not None
    ]
    if not vision_token_ids:
        return None
    vision_token_ids = list(dict.fromkeys(int(token_id) for token_id in vision_token_ids))
    return torch.isin(input_ids, torch.tensor(vision_token_ids, device=input_ids.device))


def vision_spans(vision_mask: torch.Tensor) -> list[list[tuple[int, int]]]:
    """
    Decompose the vision mask into contiguous runs of vision tokens, one list of
    (start, end) spans per batch element. Each span typically corresponds to one
    image (or one video frame block) in the prompt.
    """
    all_spans = []
    for row in vision_mask:
        idx = row.nonzero().squeeze(-1)
        spans = []
        if idx.numel() > 0:
            breaks = (idx[1:] != idx[:-1] + 1).nonzero().squeeze(-1) + 1
            starts = [0] + breaks.tolist()
            ends = breaks.tolist() + [idx.numel()]
            spans = [(int(idx[s]), int(idx[e - 1]) + 1) for s, e in zip(starts, ends)]
        all_spans.append(spans)
    return all_spans
