from nowcasting_utils.visualization.visualization import plot_example
from nowcasting_dataset.consts import NWP_VARIABLE_NAMES
from nowcasting_dataset.config.model import Configuration
from nowcasting_dataloader.fake import FakeDataset
import torch

import tilemapbase

# for Github actions need to create this
try:
    tilemapbase.init(create=True)
except Exception:
    pass


from nowcasting_dataloader.batch import BatchML


def get_batch(
):

    c = Configuration()
    c.input_data = c.input_data.set_all_to_defaults()

    # set up fake data
    train_dataset = FakeDataset(configuration=c)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=None)

    # get data
    x = next(iter(train_dataloader))
    x = BatchML(**x)

    return x


def test_plot_example():

    batch = get_batch()

    model_output = torch.randn(8, 6)

    plot_example(
        batch=batch,
        model_output=model_output,
        history_minutes=60,
        forecast_minutes=30,
        nwp_channels=NWP_VARIABLE_NAMES,
        output_variable="pv_yield",
    )


def test_plot_example_gsp_yield():

    batch = get_batch()

    model_output = torch.randn(8, 1)

    plot_example(
        batch=batch,
        model_output=model_output,
        history_minutes=60,
        forecast_minutes=30,
        nwp_channels=NWP_VARIABLE_NAMES,
        output_variable="gsp_yield",
    )
