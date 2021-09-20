from nowcasting_utils.visualization.visualization import plot_example
from nowcasting_dataset.data_sources.nwp_data_source import NWP_VARIABLE_NAMES
import torch


def get_batch(
    batch_size=8,
    seq_length_5=19,
    seq_length_30=4,
    width=64,
    height=64,
    number_sat_channels=len(NWP_VARIABLE_NAMES),
):

    x = {
        "sat_data": torch.randn(batch_size, seq_length_5, width, height, number_sat_channels),
        "pv_yield": torch.randn(batch_size, seq_length_5, 128),
        "pv_system_id": torch.randn(batch_size, 128),
        "nwp": torch.randn(batch_size, 10, seq_length_5, 2, 2),
        "hour_of_day_sin": torch.randn(batch_size, seq_length_5),
        "hour_of_day_cos": torch.randn(batch_size, seq_length_5),
        "day_of_year_sin": torch.randn(batch_size, seq_length_5),
        "day_of_year_cos": torch.randn(batch_size, seq_length_5),
        "gsp_yield": torch.randn(batch_size, seq_length_30, 32),
        "gsp_id": torch.randn(batch_size, 32),
    }

    # add a nan
    x["pv_yield"][0, 0, :] = float("nan")

    # add fake x and y coords, and make sure they are sorted
    x["sat_x_coords"], _ = torch.sort(torch.randn(batch_size, seq_length_5))
    x["sat_y_coords"], _ = torch.sort(torch.randn(batch_size, seq_length_5), descending=True)
    x["gsp_system_x_coords"], _ = torch.sort(torch.randn(batch_size, seq_length_30))
    x["gsp_system_y_coords"], _ = torch.sort(
        torch.randn(batch_size, seq_length_30), descending=True
    )

    # add sorted (fake) time series
    x["sat_datetime_index"], _ = torch.sort(torch.randn(batch_size, seq_length_5))
    x["nwp_target_time"], _ = torch.sort(torch.randn(batch_size, seq_length_5))
    x["gsp_datetime_index"], _ = torch.sort(torch.randn(batch_size, seq_length_30))

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
