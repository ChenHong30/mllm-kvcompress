# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch
from transformers.utils import logging

from mllm_kvcompress.core.runtime import CompressionMethod
from mllm_kvcompress.core.state import LayerContext

logger = logging.get_logger(__name__)


def _uniform_quantize(x: torch.Tensor, bits: float, group_size: int) -> torch.Tensor:
    """
    Simulated asymmetric uniform quantization along the last dimension in groups of
    `group_size`. Fractional bits are supported via the number of levels
    (e.g. 1.58 bits -> 3 levels). Trailing elements that do not fill a group are
    left unquantized (the official residual cache).
    """
    n_levels = max(2, round(2 ** bits))
    length = x.shape[-1]
    n_groups = length // group_size
    if n_groups == 0:
        return x
    main, residual = x[..., : n_groups * group_size], x[..., n_groups * group_size :]
    grouped = main.reshape(*main.shape[:-1], n_groups, group_size).float()
    mn = grouped.amin(dim=-1, keepdim=True)
    scale = (grouped.amax(dim=-1, keepdim=True) - mn) / (n_levels - 1)
    quantized = ((grouped - mn) / scale.clamp_min(1e-8)).round().clamp(0, n_levels - 1)
    dequantized = (quantized * scale + mn).reshape(*main.shape).to(x.dtype)
    return torch.cat([dequantized, residual], dim=-1)


@dataclass
class VidKV(CompressionMethod):
    """
    VidKV: mixed-precision ~1.x-bit KV cache quantization for video LLMs.

    VidKV pushes KV cache quantization of vision tokens below 2 bits:
    - Keys are quantized per-channel (groups along the token axis). Channels with a
      large dynamic range ("difficult", a `key_bits - 1` fraction) get 2-bit; the
      remaining channels are transformed with an FFT along the head dimension first
      and quantized at 1 bit (the FFT smooths per-channel key distributions).
    - Values are quantized per-token at 1.58 bits (3 levels), which VidKV shows is
      safe for vision tokens.
    Text KV pairs are kept in full precision.

    Based on VidKV (https://arxiv.org/abs/2503.16257),
    official implementation: https://github.com/KD-TAO/VidKV.

    Adaptation notes: this method *simulates* quantization (quantize -> dequantize
    during pre-filling) to measure its accuracy impact; the cache stays in the compute
    dtype, so no memory is actually saved -- packed low-bit storage requires the
    official triton kernels and a custom cache class. The Semantic Token Protection
    variant (keeping salient video tokens at 2 bits) is approximated by `key_bits`
    interpolation; text tokens are always protected.

    Parameters
    ----------
    key_bits : float, default=1.5
        Effective key bit-width in (1, 2]. A `key_bits - 1` fraction of channels
        (largest dynamic range) is quantized at 2 bits, the rest at 1 bit (FFT domain).
    value_bits : float, default=1.58
        Value bit-width; 1.58 corresponds to 3 quantization levels.
    group_size : int, default=32
        Quantization group size (along tokens for keys, along channels for values).
    """

    key_bits: float = 1.5
    value_bits: float = 1.58
    group_size: int = 32

    def __post_init__(self):
        assert 1 <= self.key_bits <= 2, "key_bits must be in [1, 2]"
        assert 1 <= self.value_bits <= 8, "value_bits must be in [1, 8]"

    def stats_note(self) -> str:
        return (
            f"vision keys quantized to {self.key_bits} bits, values to {self.value_bits} bits "
            "(simulated quantization: the stored cache size is unchanged)"
        )

    def compress(self, ctx: LayerContext) -> tuple[torch.Tensor, torch.Tensor]:
        if ctx.vision_mask is None:
            logger.warning_once("VidKV could not find vision token positions, no quantization is applied.")
            return ctx.keys, ctx.values

        keys, values = ctx.keys.clone(), ctx.values.clone()
        for batch_idx in range(ctx.batch_size):
            vision_positions = ctx.vision_mask[batch_idx].nonzero().squeeze(-1)
            if vision_positions.numel() < self.group_size:
                continue

            # ---- Keys: mixed 2-bit / 1-bit (FFT) per-channel quantization ----
            k_vis = keys[:, :, vision_positions][batch_idx]  # (h, n_vis, d)
            head_dim = k_vis.shape[-1]
            channel_range = k_vis.amax(dim=(0, 1)) - k_vis.amin(dim=(0, 1))  # (d,)
            n_difficult = int(head_dim * (self.key_bits - 1))

            difficult_mask = torch.zeros(head_dim, dtype=torch.bool, device=keys.device)
            if n_difficult > 0:
                difficult_mask[channel_range.topk(n_difficult).indices] = True
                # difficult channels: plain 2-bit, per-channel groups along tokens
                k_difficult = k_vis[..., difficult_mask].transpose(-2, -1)  # (h, d_diff, n_vis)
                k_difficult = _uniform_quantize(k_difficult, 2, self.group_size).transpose(-2, -1)
                k_vis[..., difficult_mask] = k_difficult

            if (~difficult_mask).any():
                # easy channels: 1-bit in the FFT domain (along the channel axis)
                k_easy = k_vis[..., ~difficult_mask]
                k_fft = torch.view_as_real(torch.fft.fft(k_easy.float(), dim=-1))  # (h, n_vis, d_easy, 2)
                k_fft = k_fft.flatten(-2).transpose(-2, -1)  # (h, 2*d_easy, n_vis)
                k_fft = _uniform_quantize(k_fft, 1, self.group_size)
                k_fft = k_fft.transpose(-2, -1).unflatten(-1, (-1, 2)).contiguous()
                k_vis[..., ~difficult_mask] = torch.fft.ifft(
                    torch.view_as_complex(k_fft), dim=-1
                ).real.to(k_vis.dtype)

            keys[batch_idx, :, vision_positions] = k_vis

            # ---- Values: per-token quantization along channels ----
            v_vis = values[batch_idx, :, vision_positions]
            values[batch_idx, :, vision_positions] = _uniform_quantize(
                v_vis, self.value_bits, min(self.group_size, v_vis.shape[-1])
            )

        return keys, values
