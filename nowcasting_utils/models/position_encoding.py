"""
This file contains various ways of performing positional encoding.

These encodings can be:
- Relative positioning (i.e. this pixel is this far from the top left, and this many timesteps in the future)
- Absolute positioning (i.e. this pixel is at this latitude/longitude, and is at 16:00)

These encodings can also be performed with:
- Fourier Features, based off what is done in PerceiverIO
- Coordinates, based off the idea from Coordinate Convolutions
"""
import numpy as np
import torch
import einops
from math import pi
from typing import Union, Optional, Dict, List, Tuple
import datetime


def encode_position(
    shape: List[..., int],
    geospatial_coordinates: Optional[Tuple[List[int, ...], List[int, ...]]],
    datetimes: Optional[List[datetime.datetime]],
    method: str,
    positioning: str,
    geospatial_bounds: Optional[List[int, int, int, int]],
    **kwargs,
) -> torch.Tensor:
    """
    This function wraps a variety of different methods for generating position features for given inputs.

    Args:
        shape: The shape of the input to be encoded, should be the largest or finest-grained input
            For example, if the inputs are shapes (12, 6, 128, 128) and (1, 6), (12, 6, 128, 128) should be passed in as
            shape, as it has the most elements and the input (1, 6) can just subselect the position encoding
        geospatial_coordinates: The latitude/longitude of the inputs for shape, in OSGB coordinates, unused if using relative positioning only
        datetimes: time of day and date for each of the timesteps in the shape, unused if using relative positioning only
        method: Method of the encoding, either 'fourier' for Fourier Features
        positioning: The type of positioning used, either 'relative' for relative positioning, or 'absolute', or 'both'
        geospatial_bounds: The bounds of the geospatial area covered, in x_min, y_min, x_max, y_max ordering, only used for absolute coordinates

    Returns:
        The position encodings for all items in the batch
    """
    assert method in [
        "fourier",
    ], ValueError(f"method must be one of 'fourier', not {method}")
    assert positioning in ["relative", "absolute", "both"], ValueError(
        f"positioning must be one of 'relative', 'absolute' or 'both', not {positioning}"
    )

    if positioning == "relative":
        position_encoding = encode_relative_position(shape, **kwargs)
    elif positioning == "absolute":
        position_encoding = encode_absolute_position(
            shape, geospatial_coordinates, geospatial_bounds, datetimes
        )
    else:
        # Both position encodings
        position_encoding = torch.cat(
            [
                encode_relative_position(shape),
                encode_absolute_position(
                    shape, geospatial_coordinates, geospatial_bounds, datetimes
                ),
            ],
            dim=-1,
        )
    return position_encoding


def encode_relative_position(shape: List[..., int], **kwargs) -> torch.Tensor:
    """
    Encode the relative position of the pixels/voxels

    Args:
        shape:

    Returns:
        The relative position encoding as a torch Tensor

    """
    position_encoding = encode_fouier_position(1, shape, **kwargs)
    return position_encoding


def encode_absolute_position(
    shape: List[..., int], geospatial_coordinates, geospatial_bounds, datetimes, **kwargs
) -> torch.Tensor:
    """
    Encodes the absolute position of the pixels/voxels in time and space

    Args:
        shape: Shape to encode positions for
        geospatial_coordinates: The geospatial coordinates, in OSGB format
        datetimes: Time of day and date as a list of datetimes, one for each timestep
        **kwargs:

    Returns:
        The absolute position encoding for the given shape
    """
    hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos = create_datetime_features(
        datetimes
    )

    pass


def normalize_geospatial_coordinates(geospatial_coordinates, geospatial_bounds) -> np.ndarray:
    """
    Normalize the geospatial coordinates by the max extant to keep everything between -1 and 1

    Args:
        geospatial_coordinates: The coordinates for the pixels in the image
        geospatial_bounds: The maximum extant

    Returns:
        The normalized geospatial coordinates, rescaled to between -1 and 1

    """
    pass


def create_datetime_features(
    datetimes: List[datetime.datetime],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a list of datetimes to day of year, hour of day sin and cos representation

    Args:
        datetimes: List of datetimes

    Returns:
        Tuple of numpy arrays containing the hour of day sin,cos, and day of year sin,cos
    """
    hour_of_day = []
    day_of_year = []
    for index in datetimes:
        hour_of_day.append((index.hour + (index.minute / 60) / 24))
        day_of_year.append((index.timetuple().tm_yday / 365))  # Get the day of the year
    hour_of_day = np.asarray(hour_of_day)
    day_of_year = np.asarray(day_of_year)
    hour_radians = hour_of_day * 2 * np.pi
    day_radians = day_of_year * 2 * np.pi
    hour_of_day_sin = np.sin(hour_radians)
    hour_of_day_cos = np.cos(hour_radians)
    day_of_year_sin = np.sin(day_radians)
    day_of_year_cos = np.cos(day_radians)

    return hour_of_day_sin, hour_of_day_cos, day_of_year_sin, day_of_year_cos


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