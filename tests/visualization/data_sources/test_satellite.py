""" Tests to plot satellite data """
import os

import plotly.graph_objects as go
from nowcasting_dataset.data_sources.fake import satellite_fake
from nowcasting_dataset.geospatial import osgb_to_lat_lon

from nowcasting_utils.visualization.data_sources.plot_satellite import (
    make_animation_all_channels,
    make_animation_one_channels,
    make_traces_one_channel,
    make_traces_one_channel_one_time,
)
from nowcasting_utils.visualization.utils import make_buttons


def test_make_traces_one_channel_one_time():
    """ Test 'make_traces_one_channel_one_time' functions """
    satellite = satellite_fake(
        batch_size=2, seq_length_5=5, satellite_image_size_pixels=32, number_satellite_channels=2
    )

    example_index = 1
    trace = make_traces_one_channel_one_time(
        satellite=satellite, example_index=example_index, channel_index=0, time_index=1
    )

    fig = go.Figure(trace)

    x = satellite.x[example_index].mean()
    y = satellite.y[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=7, mapbox_center={"lat": lat, "lon": lon}
    )

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_make_traces_one_channel():
    """ Test 'make_traces_one_channel' functions """
    satellite = satellite_fake(
        batch_size=2, seq_length_5=5, satellite_image_size_pixels=32, number_satellite_channels=2
    )

    example_index = 1
    traces = make_traces_one_channel(
        satellite=satellite, example_index=example_index, channel_index=0
    )

    x = satellite.x[example_index].mean()
    y = satellite.y[example_index].mean()

    lat, lon = osgb_to_lat_lon(x=x, y=y)

    frames = []
    for i, trace in enumerate(traces[1:]):
        frames.append(go.Frame(data=trace, name=f"frame{i+1}"))

    fig = go.Figure(
        data=traces[0],
        layout=go.Layout(
            title="Start Title",
        ),
        frames=frames,
    )
    fig.update_layout(updatemenus=[make_buttons()])
    fig.update_layout(
        mapbox_style="carto-positron", mapbox_zoom=7, mapbox_center={"lat": lat, "lon": lon}
    )

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_make_animation_one_channels():
    """ Test 'make_animation_one_channels' functions """
    satellite = satellite_fake(
        batch_size=2, seq_length_5=5, satellite_image_size_pixels=32, number_satellite_channels=2
    )

    fig = make_animation_one_channels(satellite=satellite, example_index=1, channel_index=0)

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")


def test_make_animation_all_channesl():
    """ Test 'make_animation_all_channels' functions """
    # import xarray as xr
    # satellite = xr.load_dataset('/Users/peterdudfield/Documents/Github/
    # nowcasting_utils/tests/data/sat/000000.nc')
    # satellite = satellite.rename({'stacked_eumetsat_data':'data', "variable":"channels"})

    satellite = satellite_fake(
        batch_size=2, seq_length_5=5, satellite_image_size_pixels=32, number_satellite_channels=8
    )

    fig = make_animation_all_channels(satellite=satellite, example_index=1)

    if "CI" not in os.environ.keys():
        fig.show(renderer="browser")
