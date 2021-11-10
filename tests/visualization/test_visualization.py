"""Test general visualization """
import tilemapbase
import torch
from nowcasting_dataloader.fake import FakeDataset
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataset.consts import NWP_VARIABLE_NAMES

from nowcasting_utils.visualization.visualization import plot_example

# for Github actions need to create this
try:
    tilemapbase.init(create=True)
except Exception:
    pass


from nowcasting_dataloader.batch import BatchML


def get_batch():
    """get batch for tests"""
    c = Configuration()
    c.process.batch_size = 4
    c.input_data = c.input_data.set_all_to_defaults()

    # set up fake data
    train_dataset = FakeDataset(configuration=c)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # get data
    x = next(iter(train_dataloader))
    x = BatchML(**x)

    return x


def test_b_plot_example():
    """test plot for pv yield"""
    batch = get_batch()

    model_output = torch.randn(8, 6)

    fig = plot_example(
        batch=batch,
        model_output=model_output,
        history_minutes=60,
        forecast_minutes=30,
        nwp_channels=NWP_VARIABLE_NAMES,
        output_variable="pv_yield",
    )
    fig.clear()


def test_a_plot_example_gsp_yield():
    """test plot for gsp"""
    batch = get_batch()

    model_output = torch.randn(8, 1)

    fig = plot_example(
        batch=batch,
        model_output=model_output,
        history_minutes=60,
        forecast_minutes=30,
        nwp_channels=NWP_VARIABLE_NAMES,
        output_variable="gsp_yield",
    )

    fig.clear()
