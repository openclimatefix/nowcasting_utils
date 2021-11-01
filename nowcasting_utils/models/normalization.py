"""
File containing different normalization methods
"""

import numpy as np


def metnet_normalization(data: np.ndarray) -> np.ndarray:
    """
    Perform normalization from the MetNet paper

    This involves subtracting by the median, dividing by the interquartile range,
    then squashing to [-1,1] with the hyperbolic tangent

    Args:
        data: input image data

    Returns:
        Normalized image data
    """
    # Ensure no NaNs
    data = np.nan_to_num(data)
    # Get IQR
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    data = (data - np.median(data)) / iqr
    #
    # Now hyperbolic tangent
    # data = np.tanh(data)
    # TODO tanh seems to give an issue of not forcing between -1 and 1,
    #  but gives between ~-0.76 and 0.76
    return np.clip(data, -1.0, 1.0)


def standard_normalization(data: np.ndarray, std: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """
    Performs standard normalization to get values with a mean of 0 and standard deviation of 1

    Args:
        data: The data to normalize
        std: Standard deviation of each channel
        mean: Mean of each channel

    Returns:
        The normalized data
    """
    return (data - mean) / std
