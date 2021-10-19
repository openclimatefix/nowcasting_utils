"""
Matplotlib functions to plot a example dataset and model outputs.

Author: Jack Kelly
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterable, Optional
import tilemapbase
from nowcasting_dataset.geospatial import osgb_to_lat_lon
from nowcasting_dataset.consts import GSP_ID, GSP_YIELD, DATETIME_FEATURE_NAMES


def plot_example(
    batch,
    model_output,
    history_minutes: int,
    forecast_minutes: int,
    nwp_channels: Iterable[str],
    example_i: int = 0,
    epoch: Optional[int] = None,
    output_variable: str = "pv_yield",
) -> plt.Figure:
    """
    Plots an example with the satellite imagery, timeseries and PV yield.

    Args:
        batch: The batch to plot
        model_output: The output from the model
        history_minutes: The number of minutes of the input history
        forecast_minutes: The number minutes of forecast
        nwp_channels: The names of nwp channels
        example_i: Which example to plot from the batch
        epoch: The optional epoch number
        output_variable: this can be 'pv_yield' or 'gsp_yield'

    Returns:
        Matplotlib Figure containing the plotted graphs and images
    """
    fig = plt.figure(figsize=(20, 20))
    ncols = 4
    nrows = 2

    history_len = history_minutes // 5
    forecast_len = forecast_minutes // 5

    history_len_30 = history_minutes // 30
    forecast_len_30 = forecast_minutes // 30

    # ******************* SATELLITE IMAGERY ***********************************
    extent = (  # left, right, bottom, top
        float(batch["sat_x_coords"][example_i, 0].cpu().numpy()),
        float(batch["sat_x_coords"][example_i, -1].cpu().numpy()),
        float(batch["sat_y_coords"][example_i, -1].cpu().numpy()),
        float(batch["sat_y_coords"][example_i, 0].cpu().numpy()),
    )

    def _format_ax(ax):
        if "x_meters_center" in batch:
            ax.scatter(
                batch["x_meters_center"][example_i].cpu(),
                batch["y_meters_center"][example_i].cpu(),
                s=500,
                color="white",
                marker="x",
            )

    ax = fig.add_subplot(nrows, ncols, 1)
    sat_data = batch["sat_data"][example_i, :, :, :, 0].cpu().numpy()
    sat_min = np.min(sat_data)
    sat_max = np.max(sat_data)
    ax.imshow(sat_data[0], extent=extent, interpolation="none", vmin=sat_min, vmax=sat_max)
    ax.set_title("t = -{}".format(history_len))
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 2)
    ax.imshow(
        sat_data[history_len + 1], extent=extent, interpolation="none", vmin=sat_min, vmax=sat_max
    )
    if epoch is None:
        ax.set_title("t = 0")
    else:
        ax.set_title(f"t = 0, epoch = {epoch}")
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 3)
    ax.imshow(sat_data[-1], extent=extent, interpolation="none", vmin=sat_min, vmax=sat_max)
    ax.set_title("t = {}".format(forecast_len))
    _format_ax(ax)

    ax = fig.add_subplot(nrows, ncols, 4)
    lat_lon_bottom_left = osgb_to_lat_lon(extent[0], extent[2])
    lat_lon_top_right = osgb_to_lat_lon(extent[1], extent[3])
    tiles = tilemapbase.tiles.build_OSM()
    lat_lon_extent = tilemapbase.Extent.from_lonlat(
        longitude_min=lat_lon_bottom_left[1],
        longitude_max=lat_lon_top_right[1],
        latitude_min=lat_lon_bottom_left[0],
        latitude_max=lat_lon_top_right[0],
    )
    plotter = tilemapbase.Plotter(lat_lon_extent, tile_provider=tiles, zoom=6)
    plotter.plot(ax, tiles)

    # ******************* TIMESERIES ******************************************
    # NWP
    ax = fig.add_subplot(nrows, ncols, 5)
    nwp_dt_index = pd.to_datetime(batch["nwp_target_time"][example_i].cpu().numpy(), unit="s")
    pd.DataFrame(
        batch["nwp"][example_i, :, :, 0, 0].T.cpu().numpy(),
        index=nwp_dt_index,
        columns=nwp_channels,
    ).plot(ax=ax)
    ax.set_title("NWP")

    # datetime features
    ax = fig.add_subplot(nrows, ncols, 6)
    ax.set_title("datetime features")
    datetime_features_df = pd.DataFrame(index=nwp_dt_index, columns=DATETIME_FEATURE_NAMES)
    for key in DATETIME_FEATURE_NAMES:
        datetime_features_df[key] = batch[key][example_i].cpu().numpy()
    datetime_features_df.plot(ax=ax)
    ax.legend()
    ax.set_xlabel(nwp_dt_index[0].date())

    # ************************ PV YIELD ***************************************
    if output_variable == "pv_yield":
        ax = fig.add_subplot(nrows, ncols, 7)
        ax.set_title(f"PV yield for PV ID {batch['pv_system_id'][example_i, 0].cpu()}")
        pv_actual = pd.Series(
            batch["pv_yield"][example_i, :, 0].cpu().numpy(), index=nwp_dt_index, name="actual"
        )
        pv_pred = pd.Series(
            model_output[example_i].detach().cpu().numpy(),
            index=nwp_dt_index[history_len + 1 :],
            name="prediction",
        )
        pd.concat([pv_actual, pv_pred], axis="columns").plot(ax=ax)
        ax.legend()

    if output_variable == "gsp_yield":
        ax = fig.add_subplot(nrows, ncols, 7)
        ax.set_title(f"GSP yield for GSP ID {batch[GSP_ID][example_i, 0].cpu()}")
        gsp_dt_index = pd.to_datetime(
            batch["gsp_datetime_index"][example_i].cpu().numpy(), unit="s"
        )
        gsp_actual = pd.Series(
            batch[GSP_YIELD][example_i, :, 0].cpu().numpy(), index=gsp_dt_index, name="actual"
        )
        gsp_pred = pd.Series(
            model_output[example_i].detach().cpu().numpy(),
            index=gsp_dt_index[history_len_30 + 1 :],
            name="prediction",
        )
        pd.concat([gsp_actual, gsp_pred], axis="columns").plot(ax=ax)
        ax.legend()

    return fig
