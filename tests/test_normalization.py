from nowcasting_utils.models.normalization import metnet_normalization
import numpy as np


def test_metnet_normalization():
    data = np.random.random((2, 24, 12, 256, 256))
    data *= 1000 # Ensure not already between -1 and 1
    normalized_data = metnet_normalization(data)
    assert np.isclose(np.max(normalized_data), 1.0, atol=0.001) == True
    assert np.isclose(np.min(normalized_data), -1.0, atol=0.001) == True
