import pytest
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.dataset.batch import Batch


@pytest.fixture
def batch():
    config = Configuration()
    config.input_data = config.input_data.set_all_to_defaults()
    config.process.batch_size = 4
    config.input_data.gsp.n_gsp_per_example = 16
    config.input_data.pv.n_pv_systems_per_example = 32
    return Batch.fake(configuration=config)
