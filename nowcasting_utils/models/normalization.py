import numpy as np


def metnet_normalization(data: np.ndarray) -> np.ndarray:
    """
    Perform the normalization used in the MetNet paper on the GOES data
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
    # TODO tanh seems to give an issue of not forcing between -1 and 1, but gives between ~-0.76 and 0.76
    return np.clip(data, -1., 1.)


def standard_normalization(data: np.ndarray, std: np.ndarray, mean: np.ndarray) -> np.ndarray:
    return (data - mean) / std
