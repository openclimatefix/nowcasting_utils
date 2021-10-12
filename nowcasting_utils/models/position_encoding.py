"""
This file contains various ways of performing positional encoding.

These encodings can be:
- Relative positioning (i.e. this pixel is this far from the top left, and this many timesteps in the future)
- Absolute positioning (i.e. this pixel is at this latitude/longitude, and is at 16:00)

These encodings can also be performed with:
- Fourier Features, based off what is done in PerceiverIO
- Coordinates, based off the idea from Coordinate Convolutions
"""
import torch
import einops
from math import pi
from typing import Union, Optional, Dict, List, Tuple


def encode_position(
    shape: List[..., int],
    geospatial_coordinates: Optional[Tuple[List[int, ...], List[int, ...]]],
    time_of_day: Optional[List[float]],
    day_of_year: Optional[List[float]],
    method: str,
    positioning: str,
):
    """
    This function wraps a variety of different methods for generating position features for given inputs.

    Args:
        shape: The shape of the input to be encoded, should be the largest or finest-grained input
            For example, if the inputs are shapes (12, 6, 128, 128) and (1, 6), (12, 6, 128, 128) should be passed in as
            shape, as it has the most elements and the input (1, 6) can just subselect the position encoding
        geospatial_coordinates: The latitude/longitude of the inputs for shape, unused if using relative positioning only
        time_of_day: time of day for each of the timesteps in the shape, unused if using relative positioning only
        day_of_year: Day of year for each of the timesteps in the shape, unused if using relative positioning only
        method: Method of the encoding, either 'fourier' for Fourier Features, or 'coord' for Coordinates, or 'both'
        positioning: The type of positioning used, either 'relative' for relative positioning, or 'absolute', or 'both'

    Returns:
        The position encodings for all items in the batch
    """
    assert method in ["fourier", "coord", "both"], ValueError(
        f"method must be one of 'fourier', 'coord', or 'both', not {method}"
    )
    assert positioning in ["relative", "absolute", "both"], ValueError(
        f"positioning must be one of 'relative', 'absolute'm or 'both', not {positioning}"
    )

    pass


def encode_fouier_position(
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
