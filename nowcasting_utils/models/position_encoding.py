"""
This file contains various ways of performing positional encoding, primarily:
- Relative positioning (i.e. this pixel is this far from the top left, and this many timesteps in the future)
- Absolute positioning (i.e. this pixel is at this latitude/longitude, and is at 16:00)

These encodings can also be performed with:
- Fourier Features, based off what is done in PerceiverIO
- Coordinates, based off the idea of Coordinate Convolutions
"""
import torch
import einops
from math import pi


def encode_position(
        batch_size: int,
        axis: list,
        max_frequency: float,
        num_frequency_bands: int,
        sine_only: bool = False,
) -> torch.Tensor:
    """
    Encode the Fourier Features and return them

    Args:
        batch_size: Batch size
        axis: List containing the size of each axis
        max_frequency: Max frequency
        num_frequency_bands: Number of frequency bands to use
        sine_only: (bool) Whether to only use Sine features or both Sine and Cosine, defaults to both

    Returns:
        Torch tensor containing the Fourier Features of shape [Batch, *axis]
    """
    axis_pos = list(
        map(
            lambda size: torch.linspace(-1.0, 1.0, steps=size),
            axis,
        )
    )
    pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
    enc_pos = fourier_encode(
        pos,
        max_frequency,
        num_frequency_bands,
        sine_only=sine_only,
    )
    enc_pos = einops.rearrange(enc_pos, "... n d -> ... (n d)")
    enc_pos = einops.repeat(enc_pos, "... -> b ...", b=batch_size)
    return enc_pos


def fourier_encode(
        x: torch.Tensor,
        max_freq: float,
        num_bands: int = 4,
        sine_only: bool = False,
) -> torch.Tensor:
    """
    Create Fourier Encoding

    Args:
        x: Input Torch Tensor
        max_freq: Maximum frequency for the Fourier features
        num_bands: Number of frequency bands
        sine_only: Whether to only use sine or both sine and cosine features

    Returns:
        Torch Tensor with the fourier position encoded concatenated
    """
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(
        1.0,
        max_freq / 2,
        num_bands,
        device=device,
        dtype=dtype,
        )
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = x.sin() if sine_only else torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x
